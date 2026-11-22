import json

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend import main as api
from backend.thresholds import build_validation_curve


def test_reviews_create_and_list_sqlite_audit_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "REVIEW_DB_PATH", tmp_path / "reviews.sqlite3")
    payload = {
        "applicant_ref": "demo-123",
        "probability": 0.42,
        "risk_band": "High",
        "officer_decision": "refer",
        "reason_text": "Confirm external history before proceeding.",
    }

    with TestClient(api.app) as client:
        created = client.post("/reviews", json=payload)
        listed = client.get("/reviews")
        blank_reason = client.post("/reviews", json={**payload, "reason_text": "  "})

    assert created.status_code == 201
    assert created.json()["id"] == 1
    assert created.json()["reason_text"] == payload["reason_text"]
    assert created.json()["created_at"]
    assert listed.status_code == 200
    assert listed.json()["reviews"] == [created.json()]
    assert blank_reason.status_code == 422


def test_threshold_analysis_uses_stored_validation_curve(tmp_path, monkeypatch):
    curve = build_validation_curve(
        target=np.array([0, 1, 1, 0]),
        probability=np.array([0.1, 0.4, 0.8, 0.9]),
        bins=10,
    )
    curve_path = tmp_path / "validation_curve.json"
    curve_path.write_text(json.dumps(curve), encoding="utf-8")
    monkeypatch.setattr(api, "THRESHOLD_CURVE_PATH", curve_path)

    with TestClient(api.app) as client:
        response = client.get("/threshold-analysis?fn_cost=2&fp_cost=1")

    assert response.status_code == 200
    result = response.json()
    assert result["threshold"] == 0.4
    assert result["flagged_percent"] == 75
    assert result["recall"] == 1
    assert result["precision"] == 2 / 3
    assert result["validation_sample_size"] == 4
    assert "ASSUMED" in result["assumed_cost_warning"]


def test_lookup_is_disabled_without_local_processed_data(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROCESSED_DATA_DIR", tmp_path / "missing")

    with TestClient(api.app) as client:
        response = client.get("/lookup/123")

    assert response.status_code == 404
    assert "data/processed" in response.json()["detail"]


def test_lookup_scores_held_out_feature_row_and_returns_explanations(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROCESSED_DATA_DIR", tmp_path)
    names = api.service.model_feature_names
    row = {name: 0.0 for name in names}
    row.update(SK_ID_CURR=123, TARGET=1)
    pd.DataFrame([row]).to_parquet(tmp_path / "test.parquet", index=False)
    scored = {}

    def fake_predict(matrix):
        scored["columns"] = matrix.columns.tolist()
        return np.array([0.42])

    def fake_explain(matrix, provided_features):
        scored["provided_features"] = provided_features
        return {
            "raising_risk": [{"feature": "Example", "shap_value": 0.2, "provided": True}],
            "lowering_risk": [],
            "association_note": "Model associations, not causes.",
        }

    monkeypatch.setattr(api.service, "predict_calibrated", fake_predict)
    monkeypatch.setattr(api.service, "explain_model_matrix", fake_explain)

    with TestClient(api.app) as client:
        response = client.get("/lookup/123")

    assert response.status_code == 200
    assert response.json()["applicant_ref"] == "123"
    assert response.json()["calibrated_probability"] == 0.42
    assert response.json()["top_reasons"][0]["feature"] == "Example"
    assert scored["columns"] == names
    assert all(scored["provided_features"].values())
