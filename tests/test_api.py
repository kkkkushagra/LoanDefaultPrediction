from fastapi.testclient import TestClient
import pandas as pd

from backend.main import app


client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_predict():
    response = client.post("/predict", json={
        "AMT_CREDIT": 300000,
        "AMT_INCOME_TOTAL": 100000,
        "AMT_ANNUITY": 25000,
    })
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["calibrated_probability"] <= 1
    assert body["risk_band"] in {"Low", "Medium", "High"}
    assert len(body["top_reasons"]) <= 5


def test_invalid_predict_returns_422():
    assert client.post("/predict", json={"AMT_CREDIT": "not-a-number"}).status_code == 422


def test_sensitive_field_is_not_accepted():
    assert client.post("/predict", json={"CODE_GENDER": "F"}).status_code == 422
    assert client.post("/predict", json={"DAYS_BIRTH": -10000}).status_code == 422


def test_simulation_is_monotonic_sanity():
    response = client.post("/simulate", json={
        "applicant": {
            "AMT_CREDIT": 300000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_ANNUITY": 25000,
        },
        "overrides": {"AMT_INCOME_TOTAL": 200000},
    })
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "hypothetical, not causal"
    assert body["hypothetical"]["calibrated_probability"] <= body["base"]["calibrated_probability"] + 0.01


def test_model_input_columns_match_training_feature_order():
    from backend.service import ScoringService

    service = ScoringService()
    matrix = service.transform({
        "AMT_CREDIT": 300000,
        "AMT_INCOME_TOTAL": 100000,
        "AMT_ANNUITY": 25000,
    })
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.columns.tolist() == service.features
