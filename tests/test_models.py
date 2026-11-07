from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models.training import load_model_features


def test_model_features_exclude_ids_and_target():
    frame = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "AMT_CREDIT": [100, 200],
        "CODE_GENDER": ["F", "M"],
        "DAYS_BIRTH": [-10000, -20000],
        "AGE_YEARS": [27.4, 54.8],
    })
    features = load_model_features(frame)
    assert "SK_ID_CURR" not in features
    assert "TARGET" not in features
    assert features.columns.tolist() == ["AMT_CREDIT"]


def test_top_feature_selection_fits_on_train_only(monkeypatch):
    from src.models import run

    fitted = {}

    class Selector:
        def __init__(self, **kwargs):
            pass

        def fit(self, train_x, train_y):
            fitted["columns"] = train_x.columns.tolist()
            fitted["rows"] = train_x.shape[0]
            return self

        @property
        def booster_(self):
            return self

        def feature_importance(self, importance_type):
            return np.array([9.0, 1.0])

    monkeypatch.setattr(run.lgb, "LGBMClassifier", Selector)
    train_x = pd.DataFrame({"useful": [1, 2, 3], "unused": [0, 0, 0]})
    test_x = pd.DataFrame({"useful": [99, 100], "unused": [99, 100]})

    selected_train, selected_test = run.select_top_features(
        train_x,
        pd.Series([0, 1, 0]),
        test_x,
        top_features=1,
    )

    assert fitted == {"columns": ["useful", "unused"], "rows": 3}
    assert selected_train.columns.tolist() == ["useful"]
    assert selected_test.columns.tolist() == ["useful"]


def test_tuning_trials_return_fittable_models():
    from src.models.training import tune_lightgbm, tune_xgb

    train_x = pd.DataFrame({
        "feature_a": np.arange(60, dtype=float) % 7,
        "feature_b": np.linspace(-2.0, 2.0, 60),
        "feature_c": np.sin(np.arange(60)),
    })
    train_y = pd.Series([0, 1] * 30)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for tuner in (tune_xgb, tune_lightgbm):
        model, _ = tuner(
            train_x,
            train_y,
            trials=2,
            cv=cv,
            fast=True,
            scale_pos_weight=None,
        )
        model.fit(train_x, train_y)
        predictions = model.predict(train_x)
        assert len(predictions) == len(train_x)


def test_run_module_smoke(tmp_path, monkeypatch):
    from src.models import run

    train = pd.DataFrame({
        "SK_ID_CURR": np.arange(20),
        "TARGET": [0] * 10 + [1] * 10,
        "VALUE": np.arange(20, dtype=float),
    })
    test = train.copy()
    train.to_parquet(tmp_path / "train.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "models").mkdir()
    best_name, metrics = run.main(
        train_path=tmp_path / "train.parquet",
        test_path=tmp_path / "test.parquet",
        trials=0,
        fast=True,
        top_features=None,
    )
    assert best_name
    assert Path("models/model_v1.joblib").exists()
    assert (Path("docs/model_comparison.md")).exists()


def test_run_module_command_line_entrypoint(tmp_path, monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    train = pd.DataFrame({
        "SK_ID_CURR": np.arange(20),
        "TARGET": [0] * 10 + [1] * 10,
        "VALUE": np.arange(20, dtype=float),
    })
    test = train.copy()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "processed").mkdir()
    train.to_parquet(tmp_path / "data" / "processed" / "train.parquet", index=False)
    test.to_parquet(tmp_path / "data" / "processed" / "test.parquet", index=False)
    (tmp_path / "docs").mkdir()
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(sys, "argv", ["src/models/run.py", "--trials", "0"])

    runpy.run_module("src.models.run", run_name="__main__")
