from pathlib import Path

import pandas as pd
import numpy as np

from src.models.training import load_model_features


def test_model_features_exclude_ids_and_target():
    frame = pd.DataFrame({
        "SK_ID_CURR": [1, 2],
        "TARGET": [0, 1],
        "AMT_CREDIT": [100, 200],
    })
    features = load_model_features(frame)
    assert "SK_ID_CURR" not in features
    assert "TARGET" not in features
    assert features.columns.tolist() == ["AMT_CREDIT"]


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
    )
    assert best_name
    assert Path("models/model_v1.joblib").exists()
    assert (Path("docs/model_comparison.md")).exists()
