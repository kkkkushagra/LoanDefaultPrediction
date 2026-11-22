from datetime import datetime
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, create_model

from .service import SENSITIVE_FIELDS, ScoringService
from .thresholds import analyze_validation_curve
from src.models.training import load_model_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = ROOT / "data/processed"
THRESHOLD_CURVE_PATH = ROOT / "docs/validation_threshold_curve.json"
REVIEW_DB_PATH = Path(os.environ.get("REVIEW_DB_PATH", ROOT / "data/reviews.sqlite3"))

service = ScoringService(ROOT)
preprocessor = joblib.load(ROOT / "data/processed/preprocessor.joblib")


def field_type(column: str):
    if column in {"NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE"}:
        category_index = preprocessor.categorical_columns_.index(column)
        categories = tuple(str(value) for value in preprocessor.encoder_.categories_[category_index])
        return Literal.__getitem__(categories)
    if column.startswith(("FLAG_", "REG_REGION", "REGION", "LIVE_REGION")):
        return float
    return float


application_categories = {
    column: [
        str(value)
        for value in preprocessor.encoder_.categories_[
            preprocessor.categorical_columns_.index(column)
        ]
    ]
    for column in ("NAME_CONTRACT_TYPE", "NAME_INCOME_TYPE")
}

application_fields = {
    column: (field_type(column) | None, Field(default=None))
    for column in preprocessor.input_columns_
    if column not in SENSITIVE_FIELDS
}
application_fields["SK_ID_CURR"] = (float | None, Field(default=None))
RawApplication = create_model(
    "RawApplication",
    __config__=ConfigDict(extra="forbid"),
    **application_fields,
)


class SimulateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applicant: RawApplication
    overrides: dict[str, float] = Field(default_factory=dict)


class ReviewCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applicant_ref: str = Field(min_length=1, max_length=200)
    probability: float = Field(ge=0, le=1)
    risk_band: Literal["Low", "Medium", "High"]
    officer_decision: Literal["approve", "refer", "decline"]
    reason_text: str = Field(min_length=1, max_length=2000)

    @field_validator("applicant_ref", "reason_text")
    @classmethod
    def require_non_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


app = FastAPI(title="Loan Default Calibration API")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=307)


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exception):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid input", "errors": jsonable_encoder(exception.errors())},
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    report = (ROOT / "docs/calibration_report.md").read_text(encoding="utf-8")
    return {
        "model": "xgboost_tuned",
        "version": service.artifact["version"],
        "date": service.artifact["date"],
        "metrics_documented_in": "docs/calibration_report.md",
        "assumed_cost_warning": "ASSUMED placeholders, not real bank costs.",
        "calibration_report_available": bool(report),
    }


@app.get("/application-options")
def application_options():
    return {"categories": application_categories}


def ensure_reviews_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_ref TEXT NOT NULL,
            probability REAL NOT NULL,
            risk_band TEXT NOT NULL CHECK (risk_band IN ('Low', 'Medium', 'High')),
            officer_decision TEXT NOT NULL
                CHECK (officer_decision IN ('approve', 'refer', 'decline')),
            reason_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        )
        """
    )


@app.post("/reviews", status_code=201)
def create_review(payload: ReviewCreate):
    REVIEW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(REVIEW_DB_PATH)) as connection:
        ensure_reviews_table(connection)
        cursor = connection.execute(
            """
            INSERT INTO reviews (
                applicant_ref, probability, risk_band, officer_decision, reason_text
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                payload.applicant_ref,
                payload.probability,
                payload.risk_band,
                payload.officer_decision,
                payload.reason_text,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT id, applicant_ref, probability, risk_band, officer_decision, reason_text, created_at "
            "FROM reviews WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return dict(zip(
        ("id", "applicant_ref", "probability", "risk_band", "officer_decision", "reason_text", "created_at"),
        row,
    ))


@app.get("/reviews")
def list_reviews():
    REVIEW_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(REVIEW_DB_PATH)) as connection:
        ensure_reviews_table(connection)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, applicant_ref, probability, risk_band, officer_decision, reason_text, created_at
            FROM reviews
            ORDER BY id DESC
            """
        ).fetchall()
    return {"reviews": [dict(row) for row in rows]}


@app.get("/threshold-analysis")
def threshold_analysis(
    fn_cost: float = Query(default=5.0, gt=0),
    fp_cost: float = Query(default=1.0, gt=0),
):
    if not np.isfinite(fn_cost) or not np.isfinite(fp_cost):
        raise HTTPException(status_code=422, detail="Costs must be finite and positive.")
    if not THRESHOLD_CURVE_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Stored validation predictions are unavailable; rebuild the validation threshold curve.",
        )
    curve = json.loads(THRESHOLD_CURVE_PATH.read_text(encoding="utf-8"))
    return analyze_validation_curve(curve, fn_cost, fp_cost)


@app.get("/lookup/{applicant_id}")
def lookup_applicant(applicant_id: int):
    if not PROCESSED_DATA_DIR.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Local held-out applicant lookup is unavailable: data/processed was not found.",
        )
    test_path = PROCESSED_DATA_DIR / "test.parquet"
    if not test_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Local held-out applicant data is unavailable: data/processed/test.parquet was not found.",
        )
    test_frame = pd.read_parquet(test_path)
    matches = test_frame.loc[test_frame["SK_ID_CURR"] == applicant_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Applicant was not found in the local held-out test data.")

    row = matches.iloc[[0]]
    matrix = load_model_features(row.drop(columns=["TARGET"], errors="ignore"))
    matrix = matrix.reindex(columns=service.model_feature_names).fillna(0.0)
    probability = float(service.predict_calibrated(matrix)[0])
    provided_features = {column: True for column in matrix.columns}
    explanation = service.explain_model_matrix(matrix, provided_features=provided_features)
    return {
        "applicant_ref": str(applicant_id),
        "calibrated_probability": probability,
        "risk_band": service.risk_band(probability),
        "top_reasons": explanation["raising_risk"],
        "lowering_risk": explanation["lowering_risk"],
        "association_note": explanation["association_note"],
        "prototype_warning": "Local demo only; this held-out record is not an applicant lookup service.",
    }


def decision_response(payload: dict):
    probability = service.predict(payload)
    band = service.risk_band(probability)
    reasons = service.explain(payload)["raising_risk"]
    return {
        "calibrated_probability": probability,
        "risk_band": band,
        "decision": probability >= service.cost_threshold,
        "decision_label": "above_assumed_cost_threshold",
        "cost_threshold": service.cost_threshold,
        "cost_assumption_warning": "ASSUMED placeholders, not real bank costs.",
        "top_reasons": reasons,
        "association_note": "Model associations, not causes.",
    }


@app.post("/predict")
def predict(payload: RawApplication):
    values = payload.model_dump(exclude_none=True)
    return decision_response(values)


@app.post("/explain")
def explain(payload: RawApplication):
    values = payload.model_dump(exclude_none=True)
    probability = service.predict(values)
    explanation = service.explain(values)
    return {
        "calibrated_probability": probability,
        "risk_band": service.risk_band(probability),
        "decision": probability >= service.cost_threshold,
        "top_reasons": explanation["raising_risk"],
        "lowering_risk": explanation["lowering_risk"],
        "association_note": explanation["association_note"],
    }


@app.post("/simulate")
def simulate(payload: SimulateRequest):
    allowed = {"AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"}
    unknown = set(payload.overrides) - allowed
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unsupported overrides: {sorted(unknown)}")
    base = payload.applicant.model_dump(exclude_none=True)
    response = decision_response(base)
    overridden = dict(base)
    overridden.update(payload.overrides)
    simulated = decision_response(overridden)
    return {
        "base": response,
        "hypothetical": simulated,
        "label": "hypothetical, not causal",
    }


@app.get("/portfolio/summary")
def portfolio_summary():
    return service.portfolio_summary()
