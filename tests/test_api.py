import json
from pathlib import Path

from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from backend.main import app


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_portfolio_summary_uses_documented_json_without_parquet(monkeypatch):
    def fail_if_parquet_is_read(*args, **kwargs):
        raise AssertionError("portfolio summary must not read parquet data")

    monkeypatch.setattr(pd, "read_parquet", fail_if_parquet_is_read)
    with TestClient(app) as client:
        response = client.get("/portfolio/summary")

    assert response.status_code == 200
    assert response.json() == json.loads(
        (Path(__file__).resolve().parents[1] / "docs/portfolio_summary.json").read_text(
            encoding="utf-8"
        )
    )


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


def test_valid_seven_field_application():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "AMT_CREDIT": 300000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 350000,
            "DAYS_EMPLOYED": -3653,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
        })
    assert response.status_code == 200
    assert 0 <= response.json()["calibrated_probability"] <= 1


def test_application_options_match_openapi_categories():
    with TestClient(app) as client:
        categories = client.get("/application-options").json()["categories"]
        schema = client.get("/openapi.json").json()["components"]["schemas"]["RawApplication"]["properties"]
    assert categories["NAME_CONTRACT_TYPE"] == ["Cash loans", "Revolving loans"]
    assert categories["NAME_INCOME_TYPE"] == [
        "Businessman",
        "Commercial associate",
        "Maternity leave",
        "Pensioner",
        "State servant",
        "Student",
        "Unemployed",
        "Working",
    ]
    for field, values in categories.items():
        assert schema[field]["anyOf"][0]["enum"] == values


def test_three_field_application_remains_supported():
    with TestClient(app) as client:
        response = client.post("/predict", json={
            "AMT_CREDIT": 300000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_ANNUITY": 25000,
        })
    assert response.status_code == 200


def test_invalid_category_is_rejected_by_all_scoring_endpoints():
    applicant = {
        "AMT_CREDIT": 300000,
        "AMT_INCOME_TOTAL": 100000,
        "AMT_ANNUITY": 25000,
        "NAME_CONTRACT_TYPE": "Unknown loan",
    }
    with TestClient(app) as client:
        assert client.post("/predict", json=applicant).status_code == 422
        assert client.post("/explain", json=applicant).status_code == 422
        assert client.post("/simulate", json={"applicant": applicant, "overrides": {}}).status_code == 422


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


def test_batch_transform_matches_single_applicant_transform():
    from backend.service import ScoringService

    service = ScoringService()
    applicants = pd.DataFrame([
        {
            "SK_ID_CURR": 100002,
            "AMT_CREDIT": 300000,
            "AMT_INCOME_TOTAL": 100000,
            "AMT_ANNUITY": 25000,
            "AMT_GOODS_PRICE": 350000,
            "DAYS_EMPLOYED": -3653,
            "NAME_CONTRACT_TYPE": "Cash loans",
            "NAME_INCOME_TYPE": "Working",
        },
        {
            "SK_ID_CURR": 100003,
            "AMT_CREDIT": 250000,
            "AMT_INCOME_TOTAL": 90000,
            "AMT_ANNUITY": 18000,
            "AMT_GOODS_PRICE": 225000,
            "DAYS_EMPLOYED": 0,
            "NAME_CONTRACT_TYPE": "Revolving loans",
            "NAME_INCOME_TYPE": "Pensioner",
        },
    ])
    batched = service.transform(applicants)
    single = service.transform(applicants.iloc[[0]].to_dict(orient="records")[0])
    pd.testing.assert_frame_equal(batched.iloc[[0]].reset_index(drop=True), single)


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


def test_simulation_carries_optional_application_details_to_both_scores(monkeypatch):
    from backend import main as api_module

    calls = []

    def fake_predict(payload):
        calls.append(("predict", payload.copy()))
        return 0.2

    def fake_explain(payload):
        calls.append(("explain", payload.copy()))
        return {
            "raising_risk": [],
            "lowering_risk": [],
            "association_note": "Model associations, not causes.",
        }

    monkeypatch.setattr(api_module.service, "predict", fake_predict)
    monkeypatch.setattr(api_module.service, "explain", fake_explain)
    applicant = {
        "AMT_CREDIT": 300000,
        "AMT_INCOME_TOTAL": 100000,
        "AMT_ANNUITY": 25000,
        "AMT_GOODS_PRICE": 350000,
        "DAYS_EMPLOYED": 0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
    }

    with TestClient(app) as client:
        response = client.post("/simulate", json={
            "applicant": applicant,
            "overrides": {"AMT_CREDIT": 400000},
        })

    assert response.status_code == 200
    assert {kind for kind, _ in calls} == {"predict", "explain"}
    for kind in ("predict", "explain"):
        scored = [payload for call_kind, payload in calls if call_kind == kind]
        assert len(scored) == 2
        assert scored[0] == applicant
        assert scored[1] == {**applicant, "AMT_CREDIT": 400000}


def test_manual_calibration_matches_booster_and_stored_calibrator():
    import xgboost as xgb

    from backend.service import ScoringService

    service = ScoringService()
    test = pd.DataFrame([
        {
            "AMT_CREDIT": 300000 + index * 1000,
            "AMT_INCOME_TOTAL": 100000 + index * 500,
            "AMT_ANNUITY": 25000 + index * 100,
            "AMT_GOODS_PRICE": 350000 + index * 800,
            "DAYS_EMPLOYED": -3653 - index * 30,
            "NAME_CONTRACT_TYPE": "Cash loans" if index % 2 == 0 else "Revolving loans",
            "NAME_INCOME_TYPE": "Working" if index % 2 == 0 else "Pensioner",
        }
        for index in range(20)
    ])
    frame = service.transform(test)
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