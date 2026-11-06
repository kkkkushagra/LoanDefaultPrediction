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

    def fit(self, frame: pd.DataFrame, target: pd.Series):
        frame = frame.copy()
        frame, bureau = self._engineer(frame)
        candidates = [
            column for column in [
                "DEBT_TO_INCOME",
                "ANNUITY_TO_INCOME",
                "CREDIT_TO_GOODS",
                "EMPLOYED_TO_AGE",
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
        for column in candidates:
            iv = self._iv(frame[column], target, column)
            iv_rows.append({"feature": column, "iv": iv})
            if self.iv_min <= iv <= self.iv_max:
                self.woe_maps_[column] = self._woe_map(frame[column], target, column)

        self.iv_table_ = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
        self.selected_features_ = list(self.woe_maps_)
        self.output_columns_ = self.transform(frame.iloc[: min(100, len(frame))]).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result, _ = self._engineer(frame.copy())
        numeric = result.select_dtypes(include=np.number).columns
        numeric = [c for c in numeric if c not in AUDIT_COLUMNS and c != "SK_ID_CURR" and c != "TARGET"]
        output = result[numeric].copy().astype(float)
        for column in numeric:
            output[column] = output[column].replace([np.inf, -np.inf], np.nan)
            output[column] = output[column].fillna(output[column].median()).fillna(0.0)
        for column, mapping in self.woe_maps_.items():
            if column in output:
                labels = self._bins(output[column], column)
                output[column] = labels.map(mapping).fillna(0.0)
        output = output.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return output.astype(float)

    def _engineer(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "AMT_CREDIT" in frame and "AMT_INCOME_TOTAL" in frame:
            frame["DEBT_TO_INCOME"] = safe_divide(frame["AMT_CREDIT"], frame["AMT_INCOME_TOTAL"])
        if "AMT_ANNUITY" in frame and "AMT_INCOME_TOTAL" in frame:
            frame["ANNUITY_TO_INCOME"] = safe_divide(frame["AMT_ANNUITY"], frame["AMT_INCOME_TOTAL"])
        if "AMT_CREDIT" in frame and "AMT_GOODS_PRICE" in frame:
            frame["CREDIT_TO_GOODS"] = safe_divide(frame["AMT_CREDIT"], frame["AMT_GOODS_PRICE"])
        if "DAYS_EMPLOYED" in frame and "DAYS_BIRTH" in frame:
            age = frame["DAYS_BIRTH"].abs().replace(0, np.nan)
            employed = frame["DAYS_EMPLOYED"].abs()
            frame["EMPLOYED_TO_AGE"] = safe_divide(employed.abs(), age)
        elif "DAYS_EMPLOYED" in frame and "AGE_YEARS" in frame:
            age = frame["AGE_YEARS"].replace(0, np.nan) * 365.25
            employed = frame["DAYS_EMPLOYED"].abs()
            frame["EMPLOYED_TO_AGE"] = safe_divide(employed, age)
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
            raw = pd.read_csv(path)
            numeric = raw.select_dtypes(include=np.number).columns
            numeric = [c for c in numeric if c != "SK_ID_CURR"]
            grouped = raw.groupby("SK_ID_CURR")[numeric].agg(["mean", "max", "sum"])
            grouped.columns = [f"BUREAU_{c}_{stat}" for c, stat in grouped.columns]
            bureau = grouped.reindex(frame["SK_ID_CURR"]).set_axis(frame.index).fillna(0.0)
            frame = pd.concat([frame.reset_index(drop=True), bureau.reset_index(drop=True)], axis=1).set_axis(frame.index)
            self.bureau_columns_ = bureau.columns.tolist()
        return frame, bureau

    def _bins(self, series: pd.Series, column: str):
        non_null = series.dropna()
        if len(non_null) < self.bins:
            return pd.Series(pd.cut(series, max(1, len(non_null.unique())), duplicates="drop").astype(str), index=series.index)
        return pd.Series(pd.qcut(series, self.bins, duplicates="drop").astype(str), index=series.index)

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
