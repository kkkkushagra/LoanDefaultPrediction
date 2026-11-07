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
        external_paths: dict[str, str | Path] | None = None,
        iv_min: float = 0.02,
        iv_max: float = 0.5,
        bins: int = 10,
    ):
        self.bureau_path = bureau_path
        self.external_paths = external_paths
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
        self.raw_external_columns_: list[str] = []

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
                "EXT_SOURCE_WEIGHTED_MEAN",
                "ANNUITY_TO_CREDIT",
                "EXT_SOURCE_PRODUCT_1_2",
                "EXT_SOURCE_PRODUCT_2_3",
                *bureau.columns,
            ]
            if column in frame
        ]
        candidates += [
            column for column in ["DAYS_EMPLOYED", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE"]
            if column in frame and column not in candidates
        ]
        candidates = list(dict.fromkeys(candidates))
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
        self.raw_external_columns_ = list(dict.fromkeys(self.bureau_columns_))
        self.output_columns_ = self.transform(frame.iloc[: min(100, len(frame))]).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result, _ = self._engineer(frame.copy())
        output_columns: dict[str, pd.Series] = {}
        for column in self.selected_features_:
            if column not in result:
                output_columns[f"WOE_{column}"] = pd.Series(0.0, index=result.index)
                continue
            values = result[column].replace([np.inf, -np.inf], np.nan).fillna(self.train_medians_[column])
            missing = result[column].isna() | result[column].isin([np.inf, -np.inf])
            labels = self._apply_bins(values, column)
            if missing.any():
                labels.iloc[np.flatnonzero(missing.to_numpy())] = "missing"
            output_columns[f"WOE_{column}"] = labels.map(self.woe_maps_[column]).fillna(0.0)
        for column in self.raw_external_columns_:
            if column in result:
                output_columns[f"RAW_{column}"] = result[column]
        output = pd.DataFrame(output_columns, index=result.index)
        woe_columns = [column for column in output if column.startswith("WOE_")]
        output[woe_columns] = output[woe_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
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
            weights = pd.Series({"EXT_SOURCE_1": 1.0, "EXT_SOURCE_2": 2.0, "EXT_SOURCE_3": 3.0})
            available = [column for column in source_columns if column in weights.index]
            if available:
                weighted = frame[available].mul(weights[available])
                frame["EXT_SOURCE_WEIGHTED_MEAN"] = weighted.sum(axis=1) / frame[available].notna().mul(weights[available]).sum(axis=1)
        if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(frame):
            frame["ANNUITY_TO_CREDIT"] = safe_divide(frame["AMT_ANNUITY"], frame["AMT_CREDIT"])
        if "EXT_SOURCE_PRODUCT_1_2" not in frame and {"EXT_SOURCE_1", "EXT_SOURCE_2"}.issubset(frame):
            frame["EXT_SOURCE_PRODUCT_1_2"] = frame["EXT_SOURCE_1"] * frame["EXT_SOURCE_2"]
        if "EXT_SOURCE_PRODUCT_2_3" not in frame and {"EXT_SOURCE_2", "EXT_SOURCE_3"}.issubset(frame):
            frame["EXT_SOURCE_PRODUCT_2_3"] = frame["EXT_SOURCE_2"] * frame["EXT_SOURCE_3"]

        bureau = pd.DataFrame(index=frame.index)
        if "SK_ID_CURR" not in frame:
            return frame, bureau
        paths = dict(self.external_paths or {})
        if self.bureau_path is not None:
            paths.setdefault("bureau", self.bureau_path)
        for table, configured_path in paths.items():
            path = Path(configured_path) if configured_path is not None else None
            if path is None or not path.exists():
                continue
            aggregated = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            joined = aggregated.set_index("SK_ID_CURR").reindex(frame["SK_ID_CURR"])
            joined.index = frame.index
            joined = joined.loc[:, ~joined.columns.duplicated()]
            joined.columns = [str(column) for column in joined.columns]
            frame = frame.drop(columns=joined.columns.intersection(frame.columns), errors="ignore")
            frame = pd.concat([frame, joined], axis=1)
            bureau = pd.concat([bureau, joined], axis=1)
            self.bureau_columns_.extend(joined.columns.tolist())
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
        labels.iloc[np.flatnonzero(missing.to_numpy())] = "missing"
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
            labels.iloc[np.flatnonzero(non_missing.to_numpy())] = codes.to_numpy()
        else:
            labels.iloc[np.flatnonzero(non_missing.to_numpy())] = "0"
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
