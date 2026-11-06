import numpy as np
import pandas as pd

from src.features import FeaturePipeline


def feature_frame():
    return pd.DataFrame({
        "SK_ID_CURR": range(40),
        "AMT_INCOME_TOTAL": np.linspace(20_000, 80_000, 40),
        "AMT_CREDIT": np.linspace(50_000, 300_000, 40),
        "AMT_ANNUITY": np.linspace(5_000, 20_000, 40),
        "AMT_GOODS_PRICE": np.linspace(40_000, 280_000, 40),
        "DAYS_EMPLOYED": np.linspace(-1, -3000, 40),
        "DAYS_BIRTH": np.linspace(-7000, -25000, 40),
        "EXT_SOURCE_1": np.linspace(0.1, 0.9, 40),
        "EXT_SOURCE_2": np.linspace(0.2, 0.95, 40),
        "EXT_SOURCE_3": np.linspace(0.05, 0.85, 40),
        "CODE_GENDER": ["F", "M"] * 20,
    })


def test_features_are_finite():
    frame = feature_frame()
    target = pd.Series([0, 1] * 20)
    result = FeaturePipeline(bureau_path=None).fit(frame, target).transform(frame)
    assert not result.isna().any().any()
    assert np.isfinite(result.to_numpy()).all()


def test_no_target_leakage_columns():
    frame = feature_frame()
    target = pd.Series([0, 1] * 20)
    result = FeaturePipeline(bureau_path=None).fit(frame, target).transform(frame)
    assert "TARGET" not in result.columns
    assert "CODE_GENDER" not in result.columns
    assert "DAYS_BIRTH" not in result.columns


def test_train_test_columns_match():
    frame = feature_frame()
    target = pd.Series([0, 1] * 20)
    pipeline = FeaturePipeline(bureau_path=None).fit(frame.iloc[:30], target.iloc[:30])
    train = pipeline.transform(frame.iloc[:30])
    test = pipeline.transform(frame.iloc[30:])
    assert train.columns.tolist() == test.columns.tolist()


def test_woe_train_test_values_are_consistent_and_nonzero():
    frame = feature_frame()
    target = pd.Series([0, 1] * 20)
    pipeline = FeaturePipeline(bureau_path=None).fit(frame.iloc[:30], target.iloc[:30])
    train = pipeline.transform(frame.iloc[:30])
    test = pipeline.transform(frame.iloc[30:])
    assert train.columns.tolist() == test.columns.tolist()
    assert train.notna().all().all()
    assert (train != 0).any().any()
    assert (test != 0).any().any()


def test_single_row_matches_batch_transform():
    frame = feature_frame()
    target = pd.Series([0, 1] * 20)
    pipeline = FeaturePipeline(bureau_path=None).fit(frame.iloc[:30], target.iloc[:30])
    batch = pipeline.transform(frame.iloc[30:])
    for row in range(len(frame.iloc[30:])):
        pd.testing.assert_series_equal(
            pipeline.transform(frame.iloc[[30 + row]]).iloc[0],
            batch.iloc[row],
            check_names=False,
        )


def test_iv_dropped_columns_are_absent():
    frame = feature_frame()
    target = pd.Series([0] * 40)
    result = FeaturePipeline(bureau_path=None, iv_min=10).fit(frame, target).transform(frame)
    assert result.empty or list(result.columns) == []


def test_rare_flag_keeps_positive_values():
    frame = feature_frame()
    frame["RARE_FLAG"] = [1] + [0] * 39
    target = pd.Series([0, 1] + [0] * 38)
    result = FeaturePipeline(bureau_path=None).fit(frame, target).transform(frame)
    pipeline = FeaturePipeline(bureau_path=None).fit(frame.iloc[:30], target.iloc[:30])
    transformed = pipeline.transform(frame.iloc[30:])
    assert len(transformed) == 10


def test_external_aggregate_join_is_train_only(tmp_path):
    frame = feature_frame().iloc[:30].copy()
    external = pd.DataFrame({
        "SK_ID_CURR": range(30),
        "EXTERNAL_TOTAL": np.linspace(1, 30, 30),
    }).assign(SK_ID_CURR=lambda value: value["SK_ID_CURR"])
    path = tmp_path / "external.parquet"
    external.to_parquet(path, index=False)
    pipeline = FeaturePipeline(
        bureau_path=None,
        external_paths={"external": path},
    ).fit(frame, pd.Series([0, 1] * 15))
    transformed = pipeline.transform(frame)
    assert transformed.shape[0] == len(frame)
