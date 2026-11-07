import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.models.explain import explain_applicant
from src.models.training import load_model_features


SENSITIVE_FIELDS = {"CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScoringService:
    def __init__(self, root: Path = PROJECT_ROOT):
        self.root = root
        self.artifact = joblib.load(root / "models/model_v2.joblib")
        self.preprocessor = joblib.load(root / "data/processed/preprocessor.joblib")
        self.feature_pipeline = joblib.load(root / "data/processed/feature_pipeline.joblib")
        self.features = self.artifact["features"]
        self.base_model = self.artifact["base_model"]
        self.calibrated_model = self.artifact["calibrated_model"]
        self.lower_risk_threshold, self.upper_risk_threshold = self.artifact["risk_thresholds"]
        self.cost_threshold = float(self.artifact["cost_threshold"])

    def transform(self, payload: dict) -> pd.DataFrame:
        frame = pd.DataFrame([payload])
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
        matrix = normalized.reindex(columns=self.features).fillna(0.0)
        return matrix

    def predict(self, payload: dict) -> float:
        matrix = self.transform(payload)
        return float(self.calibrated_model.predict_proba(matrix)[:, 1][0])

    def explain(self, payload: dict) -> dict:
        matrix = self.transform(payload)
        return explain_applicant(self.base_model, matrix, row=0)

    def risk_band(self, probability: float) -> str:
        if probability < self.lower_risk_threshold:
            return "Low"
        if probability < self.upper_risk_threshold:
            return "Medium"
        return "High"

    def portfolio_summary(self) -> dict:
        path = self.root / "docs/portfolio_summary.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        probabilities = self.calibrated_model.predict_proba(
            load_model_features(
                pd.read_parquet(self.root / "data/processed/test.parquet").drop(columns=["TARGET"], errors="ignore")
            ).reindex(columns=self.features).fillna(0.0)
        )[:, 1]
        target = pd.read_parquet(self.root / "data/processed/test.parquet")["TARGET"]
        bands = np.array([self.risk_band(value) for value in probabilities])
        return {
            "rows": [
                {
                    "band": band,
                    "count": int((bands == band).sum()),
                    "applicant_share": float((bands == band).mean()),
                    "observed_default_rate": float(target.to_numpy()[bands == band].mean()),
                }
                for band in ("Low", "Medium", "High")
            ]
        }
