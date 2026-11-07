from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from .service import SENSITIVE_FIELDS, ScoringService


ROOT = Path(__file__).resolve().parents[1]

service = ScoringService(ROOT)
preprocessor = joblib.load(ROOT / "data/processed/preprocessor.joblib")


def field_type(column: str):
    if column.startswith(("FLAG_", "REG_REGION", "REGION", "LIVE_REGION")):
        return float
    return float


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
    applicant: dict[str, Any]
    overrides: dict[str, float] = Field(default_factory=dict)


app = FastAPI(title="Loan Default Calibration API")


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exception):
    return JSONResponse(status_code=422, content={"detail": "Invalid input", "errors": exception.errors()})


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
    base = dict(payload.applicant)
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
