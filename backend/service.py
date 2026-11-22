import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.explain import explain_applicant
from src.models.training import load_model_features
from .thresholds import (
    DEFAULT_FN_COST,
    DEFAULT_FP_COST,
    analyze_validation_curve,
    load_validation_curve,
)


SENSITIVE_FIELDS = {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DERIVED_FEATURE_INPUTS = {
    "ANNUITY_TO_CREDIT": {"AMT_ANNUITY", "AMT_CREDIT"},
    "ANNUITY_TO_INCOME": {"AMT_ANNUITY", "AMT_INCOME_TOTAL"},
    "CREDIT_TO_GOODS": {"AMT_CREDIT", "AMT_GOODS_PRICE"},
    "DEBT_TO_INCOME": {"AMT_CREDIT", "AMT_INCOME_TOTAL"},
    "EXT_SOURCE_PRODUCT_1_2": {"EXT_SOURCE_1", "EXT_SOURCE_2"},
    "EXT_SOURCE_PRODUCT_2_3": {"EXT_SOURCE_2", "EXT_SOURCE_3"},
}
EXT_SOURCE_AGGREGATES = {
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_MIN",
    "EXT_SOURCE_MAX",
    "EXT_SOURCE_WEIGHTED_MEAN",
}


class ScoringService:
    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root
        self.artifact = joblib.load(root / "models/model_v2.joblib")
        self.preprocessor = joblib.load(root / "data/processed/preprocessor.joblib")
        self.feature_pipeline = joblib.load(root / "data/processed/feature_pipeline.joblib")
        self.features = self.artifact["features"]
        self.base_model = self.artifact["base_model"]
        self.calibrated_model = self.artifact["calibrated_model"]
        self.xgb_model = self._find_xgb_model(self.base_model)
        feature_names = self.xgb_model.get_booster().feature_names
        if not feature_names:
            raise RuntimeError("Underlying xgboost model has no fitted booster feature names")
        self.model_feature_names = list(feature_names)
        self.lower_risk_threshold, self.upper_risk_threshold = self.artifact["risk_thresholds"]
        try:
            self.predict_calibrated(self._to_model_frame(pd.DataFrame([[np.nan] * len(self.model_feature_names)])))
        except Exception as exc:
            raise RuntimeError(f"Model startup self-check failed: {exc}") from exc

    @property
    def cost_threshold(self) -> float:
        curve = load_validation_curve(self.root / "docs/validation_threshold_curve.json")
        analysis = analyze_validation_curve(curve, DEFAULT_FN_COST, DEFAULT_FP_COST)
        return float(analysis["threshold"])

    @staticmethod
    def _find_xgb_model(value):
        import xgboost as xgb

        if isinstance(value, xgb.XGBClassifier):
            return value
        if hasattr(value, "estimator"):
            found = ScoringService._find_xgb_model(value.estimator)
            if found is not None:
                return found
        if hasattr(value, "calibrated_classifiers_"):
            for fitted in value.calibrated_classifiers_:
                found = ScoringService._find_xgb_model(fitted.estimator)
                if found is not None:
                    return found
        return None

    def _to_model_frame(self, values) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        elif isinstance(values, dict):
            frame = pd.DataFrame([values])
        else:
            frame = pd.DataFrame(values)
        return frame.reindex(columns=self.model_feature_names).astype(float)

    def predict_calibrated(self, matrix: pd.DataFrame) -> np.ndarray:
        # xgboost 3.1.x cannot read column names from pandas 3 DataFrames, so build the
        # DMatrix explicitly with the booster's own feature names.
        frame = self._to_model_frame(matrix)
        dmatrix = xgb.DMatrix(frame.to_numpy(dtype=float), feature_names=self.model_feature_names)
        raw_probability = self.xgb_model.get_booster().predict(dmatrix)
        fitted = self.calibrated_model.calibrated_classifiers_[0]
        calibrator = fitted.calibrators[0]
        if hasattr(calibrator, "predict_proba"):
            calibrated = calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        else:
            calibrated = np.asarray(calibrator.predict(raw_probability))
        return calibrated

    def transform(self, payload: dict | pd.DataFrame) -> pd.DataFrame:
        frame = payload.copy() if isinstance(payload, pd.DataFrame) else pd.DataFrame([payload])
        missing_columns = {
            column: np.nan
            for column in self.preprocessor.input_columns_
            if column not in frame
        }
        if missing_columns:
            frame = pd.concat([frame, pd.DataFrame(missing_columns, index=frame.index)], axis=1)
        frame["SK_ID_CURR"] = frame["SK_ID_CURR"].fillna(-1) if "SK_ID_CURR" in frame else -1

        clean_columns = [column for column in self.preprocessor.input_columns_ if column not in SENSITIVE_FIELDS]
        application = frame.reindex(columns=clean_columns).copy()
        if "DAYS_EMPLOYED" in application:
            sentinel = application["DAYS_EMPLOYED"].eq(365243)
            application.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
            application["DAYS_EMPLOYED_ANOMALY"] = sentinel.astype("int8")
        processed = self.preprocessor.transform(application)
        engineered = self.feature_pipeline.transform(frame.reindex(columns=clean_columns + ["SK_ID_CURR"]))
        matrix = pd.concat([processed.reset_index(drop=True), engineered.reset_index(drop=True)], axis=1)
        normalized = load_model_features(matrix)
        return self._to_model_frame(normalized.fillna(0.0))

    def predict(self, payload: dict) -> float:
        matrix = self.transform(payload)
        return float(self.predict_calibrated(self._to_model_frame(matrix))[0])

    def explain(self, payload: dict) -> dict:
        matrix = self.transform(payload)
        provided_fields = {
            field for field, value in payload.items()
            if value is not None
        }
        provided_features = {
            column: self._feature_was_provided(column, provided_fields)
            for column in matrix.columns
        }
        return self.explain_model_matrix(matrix, provided_features)

    def explain_model_matrix(
        self,
        matrix: pd.DataFrame,
        provided_features: dict[str, bool] | None = None,
    ) -> dict:
        return explain_applicant(
            self.base_model,
            self._to_model_frame(matrix),
            row=0,
            provided_features=provided_features,
        )

    def _feature_was_provided(self, column: str, provided_fields: set[str]) -> bool:
        if column.startswith("RAW_"):
            return False

        feature = column.removeprefix("WOE_")
        if feature == "DAYS_EMPLOYED_ANOMALY":
            return "DAYS_EMPLOYED" in provided_fields
        if feature in DERIVED_FEATURE_INPUTS:
            return DERIVED_FEATURE_INPUTS[feature].issubset(provided_fields)
        if feature in EXT_SOURCE_AGGREGATES:
            return bool({"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"} & provided_fields)

        return any(
            field in provided_fields
            and (feature == field or feature.startswith(f"{field}_"))
            for field in self.preprocessor.input_columns_
        )

    def risk_band(self, probability: float) -> str:
        if probability < self.lower_risk_threshold:
            return "Low"
        if probability < self.upper_risk_threshold:
            return "Medium"
        return "High"

    def portfolio_summary(self) -> dict:
        path = self.root / "docs/portfolio_summary.json"
        return json.loads(path.read_text(encoding="utf-8"))
