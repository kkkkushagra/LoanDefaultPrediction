import pandas as pd
import pytest

from src.data.loading import load_application_data
from src.data.pipeline import DataPreprocessor, prepare_data
from src.data.preprocessing import clean_application_data, extract_audit_frame


@pytest.fixture
def application_frame():
    return pd.DataFrame({
        "SK_ID_CURR": range(20),
        "TARGET": [0, 1] * 10,
        "AMT_INCOME_TOTAL": [1000, 2000] * 9 + [3000, 999999],
        "CODE_GENDER": ["M", "F"] * 10,
        "NAME_CONTRACT_TYPE": ["Cash"] * 10 + ["Revolving"] * 10,
        "DAYS_EMPLOYED": [None, 1] * 10,
    })


def test_load_validates_required_columns(tmp_path):
    frame = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [1]})
    path = tmp_path / "application.csv"
    frame.to_csv(path, index=False)
    assert load_application_data(path)["SK_ID_CURR"].tolist() == [1]


def test_validation_rejects_invalid_target(tmp_path):
    path = tmp_path / "application.csv"
    pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [2]}).to_csv(path, index=False)
    with pytest.raises(Exception):
        load_application_data(path)


def test_clean_excludes_sensitive_columns(application_frame):
    features, target = clean_application_data(application_frame)
    assert "CODE_GENDER" not in features
    assert "SK_ID_CURR" not in features
    assert target.equals(application_frame["TARGET"])


def test_preprocessor_fits_training_stats_only(application_frame):
    features, _ = clean_application_data(application_frame)
    train = features.iloc[:10].copy()
    test = features.iloc[10:].copy()
    train.loc[:, "AMT_INCOME_TOTAL"] = train["AMT_INCOME_TOTAL"].clip(
        upper=train["AMT_INCOME_TOTAL"].quantile(0.99)
    )
    preprocessor = DataPreprocessor().fit(train)
    transformed = preprocessor.transform(test)
    assert preprocessor.numeric_medians_["AMT_INCOME_TOTAL"] == pytest.approx(
        train["AMT_INCOME_TOTAL"].median()
    )
    assert not transformed.isna().any().any()


def test_prepare_data_splits_and_transforms(tmp_path, application_frame):
    path = tmp_path / "application.csv"
    application_frame.to_csv(path, index=False)
    train_x, test_x, train_y, test_y = prepare_data(path)
    assert len(train_x) == len(train_y) == 16
    assert len(test_x) == len(test_y) == 4
    assert not train_x.isna().any().any()
    assert not test_x.isna().any().any()


def test_unseen_category_is_encoded(tmp_path, application_frame):
    features, target = clean_application_data(application_frame)
    preprocessor = DataPreprocessor().fit(features.iloc[:15])
    transformed = preprocessor.transform(features.iloc[15:].assign(NAME_CONTRACT_TYPE="Unknown"))
    assert not transformed.isna().any().any()


def test_missing_columns_are_added_in_transform(application_frame):
    features, _ = clean_application_data(application_frame)
    preprocessor = DataPreprocessor().fit(features)
    transformed = preprocessor.transform(features.drop(columns=["DAYS_EMPLOYED"]))
    assert "DAYS_EMPLOYED" in preprocessor.numeric_columns_
    assert not transformed.isna().any().any()


def test_days_employed_sentinel_creates_anomaly(application_frame):
    frame = application_frame.copy()
    frame.loc[0, "DAYS_EMPLOYED"] = 365243
    features, _ = clean_application_data(frame)
    assert features.loc[0, "DAYS_EMPLOYED_ANOMALY"] == 1
    assert pd.isna(features.loc[0, "DAYS_EMPLOYED"])


def test_missing_indicators_are_train_learned(application_frame):
    features, _ = clean_application_data(application_frame)
    features["SPARSE_FIELD"] = [None] * 12 + list(range(8))
    preprocessor = DataPreprocessor().fit(features.iloc[:15])
    assert "SPARSE_FIELD" in preprocessor.missing_indicator_columns_


def test_audit_columns_are_separate(application_frame):
    audit = extract_audit_frame(application_frame)
    features, _ = clean_application_data(application_frame)
    assert {"CODE_GENDER", "AGE_YEARS"}.issubset(audit.columns)
    assert "CODE_GENDER" not in features
    assert "DAYS_BIRTH" not in features


def test_binary_and_flag_columns_are_not_clipped():
    frame = pd.DataFrame({
        "FLAG_TEST": [0, 1] * 10,
        "BINARY_VALUE": [0, 1] * 10,
        "AMOUNT": range(20),
    })
    preprocessor = DataPreprocessor(outlier_quantiles=(0.01, 0.99)).fit(frame)
    result = preprocessor.transform(frame)
    assert set(result["FLAG_TEST"].unique()).issubset({0.0, 1.0})
    assert set(result["BINARY_VALUE"].unique()).issubset({0.0, 1.0})


def test_audit_join_is_preserved_on_application_id(application_frame):
    audit = extract_audit_frame(application_frame.assign(SK_ID_CURR=range(20)))
    assert audit["SK_ID_CURR"].tolist() == application_frame.index.tolist()
