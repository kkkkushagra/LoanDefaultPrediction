from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


AUDIT_COLUMNS = {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"}
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan)


class FeaturePipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        bureau_path: str | Path | None = "data/bureau.csv",
        iv_min: float = 0.02,
        iv_max: float = 0.5,
        bins: int = 10,
    ):
        self.bureau_path = bureau_path
        self.iv_min = iv_min
        self.iv_max = iv_max
        self.bins = bins
        self.iv_table_: pd.DataFrame = pd.DataFrame()
        self.selected_features_: list[str] = []
        self.output_columns_: list[str] = []
        self.woe_maps_: dict[str, dict[str, float]] = {}
        self.bureau_columns_: list[str] = []
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.train_medians_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, target: pd.Series):
        frame = frame.copy()
        frame, bureau = self._engineer(frame)
        candidates = [
            column for column in [
                "DEBT_TO_INCOME",
                "ANNUITY_TO_INCOME",
                "CREDIT_TO_GOODS",
                "EXT_SOURCE_MEAN",
                "EXT_SOURCE_MIN",
                "EXT_SOURCE_MAX",
                *bureau.columns,
            ]
            if column in frame
        ]
        candidates += [
            column for column in ["DAYS_EMPLOYED", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE"]
            if column in frame and column not in candidates
        ]
        iv_rows = []
        self.woe_maps_ = {}
        self.bin_edges_ = {}
        self.train_medians_ = {}
        for column in candidates:
            cleaned = frame[column].replace([np.inf, -np.inf], np.nan)
            self.train_medians_[column] = float(cleaned.dropna().median()) if cleaned.notna().any() else 0.0
            imputed = cleaned.fillna(self.train_medians_[column])
            self.bin_edges_[column] = self._fit_edges(imputed)
            iv = self._iv(frame[column], target, column)
            decision = "drop" if iv < self.iv_min else "review" if iv > self.iv_max else "retain"
            iv_rows.append({"feature": column, "iv": iv, "decision": decision})
            if decision != "drop":
                self.woe_maps_[column] = self._woe_map(frame[column], target, column)

        self.iv_table_ = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
        self.selected_features_ = list(self.woe_maps_)
        self.output_columns_ = self.transform(frame.iloc[: min(100, len(frame))]).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result, _ = self._engineer(frame.copy())
        output = pd.DataFrame(index=result.index)
        for column in self.selected_features_:
            if column not in result:
                output[f"WOE_{column}"] = 0.0
                continue
            values = result[column].replace([np.inf, -np.inf], np.nan).fillna(self.train_medians_[column])
            missing = result[column].isna() | result[column].isin([np.inf, -np.inf])
            labels = self._apply_bins(values, column)
            if missing.any():
                labels.loc[missing] = "missing"
            output[f"WOE_{column}"] = labels.map(self.woe_maps_[column]).fillna(0.0)
        output = output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return output.astype(float)

    def _engineer(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "AMT_CREDIT" in frame and "AMT_INCOME_TOTAL" in frame:
            frame["DEBT_TO_INCOME"] = safe_divide(frame["AMT_CREDIT"], frame["AMT_INCOME_TOTAL"])
        if "AMT_ANNUITY" in frame and "AMT_INCOME_TOTAL" in frame:
            frame["ANNUITY_TO_INCOME"] = safe_divide(frame["AMT_ANNUITY"], frame["AMT_INCOME_TOTAL"])
        if "AMT_CREDIT" in frame and "AMT_GOODS_PRICE" in frame:
            frame["CREDIT_TO_GOODS"] = safe_divide(frame["AMT_CREDIT"], frame["AMT_GOODS_PRICE"])
        source_columns = [c for c in EXT_SOURCE_COLUMNS if c in frame]
        if source_columns:
            sources = frame[source_columns]
            frame["EXT_SOURCE_MEAN"] = sources.mean(axis=1)
            frame["EXT_SOURCE_MIN"] = sources.min(axis=1)
            frame["EXT_SOURCE_MAX"] = sources.max(axis=1)

        bureau = pd.DataFrame(index=frame.index)
        if self.bureau_path is None:
            return frame, bureau
        path = Path(self.bureau_path)
        if "SK_ID_CURR" in frame and path.exists():
            aggregated = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            bureau = aggregated.set_index("SK_ID_CURR").reindex(frame["SK_ID_CURR"])
            bureau.index = frame.index
            frame = pd.concat([frame.reset_index(drop=True), bureau.reset_index(drop=True)], axis=1).set_axis(frame.index)
            self.bureau_columns_ = bureau.columns.tolist()
        return frame, bureau

    def _fit_edges(self, series: pd.Series) -> np.ndarray:
        unique = series.dropna().unique()
        if len(unique) <= 1:
            return np.array([], dtype=float)
        quantiles = np.linspace(0, 1, self.bins + 1)
        edges = np.unique(np.quantile(series.dropna(), quantiles)[1:-1])
        return edges.astype(float)

    def _apply_bins(self, series: pd.Series, column: str) -> pd.Series:
        labels = pd.Series("-2", index=series.index, dtype=object)
        missing = series.isna() | series.isin([np.inf, -np.inf])
        labels.loc[missing] = "missing"
        non_missing = ~missing
        edges = self.bin_edges_[column]
        if len(edges):
            cut_edges = np.concatenate(([-np.inf], edges, [np.inf]))
            codes = pd.cut(
                series.loc[non_missing],
                bins=cut_edges,
                labels=False,
                include_lowest=True,
            ).astype(int).astype(str)
            labels.loc[non_missing] = codes
        else:
            labels.loc[non_missing] = "0"
        return labels

    def _bins(self, series: pd.Series, column: str):
        values = series.replace([np.inf, -np.inf], np.nan)
        return self._apply_bins(values, column)

    def _iv(self, series: pd.Series, target: pd.Series, column: str) -> float:
        grouped = pd.DataFrame({"bin": self._bins(series, column), "target": target.to_numpy()})
        counts = grouped.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        goods = counts["count"] - counts["sum"]
        goods = goods.clip(lower=1e-6)
        bads = counts["sum"].clip(lower=1e-6)
        goods_share = goods / goods.sum()
        bads_share = bads / bads.sum()
        return float(((goods_share - bads_share) * np.log(goods_share / bads_share)).sum())

    def _woe_map(self, series: pd.Series, target: pd.Series, column: str) -> dict[str, float]:
        grouped = pd.DataFrame({"bin": self._bins(series, column), "target": target.to_numpy()})
        counts = grouped.groupby("bin", observed=True)["target"].agg(["count", "sum"])
        goods = (counts["count"] - counts["sum"]).clip(lower=1e-6)
        bads = counts["sum"].clip(lower=1e-6)
        return np.log((goods / goods.sum()) / (bads / bads.sum())).to_dict()
