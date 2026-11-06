import pandas as pd
import pytest

from src.data.loading import load_application_data
from src.data.pipeline import DataPreprocessor, prepare_data
from src.data.preprocessing import clean_application_data


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
