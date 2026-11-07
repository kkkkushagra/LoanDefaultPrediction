from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from backend.main import app


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_valid_predict():
    with TestClient(app) as client:
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
    with TestClient(app) as client:
        assert client.post("/predict", json={"AMT_CREDIT": "not-a-number"}).status_code == 422


def test_sensitive_field_is_not_accepted():
    with TestClient(app) as client:
        assert client.post("/predict", json={"CODE_GENDER": "F"}).status_code == 422
        assert client.post("/predict", json={"DAYS_BIRTH": -10000}).status_code == 422


def test_simulation_is_monotonic_sanity():
    with TestClient(app) as client:
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
    assert matrix.columns.tolist() == service.model_feature_names
    assert service.model_feature_names == list(service.xgb_model.get_booster().feature_names)


def test_explain_endpoint():
    with TestClient(app) as client:
        response = client.post("/explain", json={
            "AMT_CREDIT": 300000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_ANNUITY": 25000,
        })
        assert response.status_code == 200
        assert "lowering_risk" in response.json()


def test_higher_credit_simulation():
    with TestClient(app) as client:
        response = client.post("/simulate", json={
            "applicant": {
                "AMT_CREDIT": 300000,
                "AMT_INCOME_TOTAL": 100000,
                "AMT_ANNUITY": 25000,
            },
            "overrides": {"AMT_CREDIT": 500000},
        })
        assert response.status_code == 200
        assert response.json()["label"] == "hypothetical, not causal"


def test_manual_calibration_matches_wrapper_on_test_rows():
    import xgboost as xgb

    from backend.service import ScoringService
    from src.models.training import load_model_features

    service = ScoringService()
    test = pd.read_parquet("data/processed/test.parquet").drop(columns=["TARGET"], errors="ignore").iloc[:20]
    matrix = load_model_features(test).reindex(columns=service.model_feature_names).fillna(0.0)
    frame = service._to_model_frame(matrix)
    manual = service.predict_calibrated(frame)

    # xgboost 3.1 cannot read column names from pandas 3 DataFrames, so the sklearn wrapper
    # cannot be called directly. Verify against the booster + stored calibrator instead.
    booster = service.xgb_model.get_booster()
    dmatrix = xgb.DMatrix(frame.to_numpy(dtype=float), feature_names=service.model_feature_names)
    raw = booster.predict(dmatrix)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    assert np.allclose(1.0 / (1.0 + np.exp(-contributions.sum(axis=1))), raw, atol=1e-4)

    calibrator = service.calibrated_model.calibrated_classifiers_[0].calibrators[0]
    if hasattr(calibrator, "predict_proba"):
        expected = calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        expected = np.asarray(calibrator.predict(raw))
    assert np.max(np.abs(expected - manual)) <= 1e-6
    assert ((manual >= 0.0) & (manual <= 1.0)).all()