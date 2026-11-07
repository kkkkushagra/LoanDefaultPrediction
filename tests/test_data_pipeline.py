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


def test_external_aggregates(tmp_path):
    from scripts.aggregate_external_data import (
        bureau_aggregates,
        installment_aggregates,
        previous_application_aggregates,
    )

    bureau = pd.DataFrame({"SK_ID_CURR": [1, 1], "AMT_CREDIT_SUM": [100.0, 200.0]})
    previous = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "AMT_APPLICATION": [100.0, 200.0],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
    })
    installments = pd.DataFrame({
        "SK_ID_CURR": [1, 1],
        "AMT_PAYMENT": [100.0, 100.0],
        "DAYS_INSTALMENT": [10, 20],
        "DAYS_ENTRY_PAYMENT": [11, 19],
    })

    bureau_result = bureau_aggregates(bureau)
    assert "BUREAU_AMT_CREDIT_SUM_MEAN_AGG" in bureau_result.columns
    assert "BUREAU_ROW_COUNT" in bureau_result.columns
    assert "SUM_AGG" not in bureau_aggregates(bureau).columns
    assert previous_application_aggregates(previous)["PREV_APPROVED_SHARE"].iloc[0] == 0.5
    assert installment_aggregates(installments)["INSTALL_LATE_PAYMENT_RATE"].iloc[0] == 0.5


def test_external_aggregates_real_table_shapes():
    from scripts.aggregate_external_data import (
        bureau_aggregates,
        installment_aggregates,
        previous_application_aggregates,
    )

    bureau = pd.DataFrame({
        "SK_ID_CURR": [100001, 100001, 100002],
        "SK_ID_BUREAU": [5001, 5002, 5003],
        "DAYS_CREDIT": [-100, -200, -50],
        "AMT_CREDIT_SUM": [1000.0, 2000.0, 3000.0],
        "AMT_CREDIT_SUM_DEBT": [100.0, 200.0, 300.0],
    })
    previous = pd.DataFrame({
        "SK_ID_PREV": [2001, 2002, 2003],
        "SK_ID_CURR": [100001, 100001, 100002],
        "NAME_CONTRACT_STATUS": ["Approved", "Refused", "Approved"],
        "AMT_APPLICATION": [1000.0, 2000.0, 3000.0],
        "AMT_CREDIT": [900.0, 1900.0, 2900.0],
    })
    installments = pd.DataFrame({
        "SK_ID_PREV": [2001, 2001, 2002],
        "SK_ID_CURR": [100001, 100001, 100002],
        "NUM_INSTALMENT_VERSION": [1.0, 1.0, 2.0],
        "NUM_INSTALMENT_NUMBER": [1.0, 2.0, 1.0],
        "DAYS_INSTALMENT": [-30.0, -10.0, -20.0],
        "DAYS_ENTRY_PAYMENT": [-29.0, -12.0, -20.0],
        "AMT_INSTALMENT": [500.0, 500.0, 750.0],
        "AMT_PAYMENT": [500.0, 490.0, 750.0],
    })

    bureau_result = bureau_aggregates(bureau)
    previous_result = previous_application_aggregates(previous)
    installment_result = installment_aggregates(installments)

    assert "SK_ID_BUREAU" not in "".join(bureau_result.columns)
    assert previous_result["PREV_ROW_COUNT"].tolist() == [2, 1]
    assert "SK_ID_PREV" not in "".join(previous_result.columns)
    assert "INSTALL_LATE_PAYMENT_RATE" in installment_result


def test_balance_external_aggregates():
    from scripts.aggregate_external_data import (
        credit_card_aggregates,
        pos_cash_aggregates,
    )

    pos_cash = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [10, 10, 20],
        "SK_DPD": [0, 5, 0],
        "SK_DPD_DEF": [0, 1, 0],
        "CNT_INSTALMENT_FUTURE": [2, 1, 3],
    })
    credit_card = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_PREV": [10, 10, 20],
        "AMT_BALANCE": [100.0, 50.0, 0.0],
        "AMT_CREDIT_LIMIT_ACTUAL": [200.0, 200.0, 100.0],
        "SK_DPD": [0, 1, 0],
        "SK_DPD_DEF": [0, 0, 0],
    })

    pos_result = pos_cash_aggregates(pos_cash)
    card_result = credit_card_aggregates(credit_card)

    assert "POS_SK_DPD_LATE_RATE" in pos_result
    assert pos_result.loc[1, "POS_SK_DPD_LATE_RATE"] == 0.5
    assert "CREDIT_CARD_UTILISATION_MEAN" in card_result
    assert card_result.loc[1, "CREDIT_CARD_UTILISATION_MEAN"] == pytest.approx(0.375)


@pytest.mark.parametrize(
    ("aggregator_name", "frame", "expected_column"),
    [
        (
            "bureau_aggregates",
            lambda: pd.DataFrame({
                "SK_ID_CURR": [1, 1],
                "SK_ID_BUREAU": [10, 11],
                "CREDIT_ACTIVE": ["Active", "Closed"],
                "AMT_CREDIT_SUM": [100.0, 200.0],
                "AMT_CREDIT_SUM_DEBT": [20.0, 0.0],
                "TARGET": [0, 1],
            }),
            "BUREAU_ACTIVE_DEBT_CREDIT_RATIO",
        ),
        (
            "previous_application_aggregates",
            lambda: pd.DataFrame({
                "SK_ID_CURR": [1, 1],
                "SK_ID_PREV": [20, 21],
                "AMT_APPLICATION": [100.0, 200.0],
                "AMT_CREDIT": [90.0, 180.0],
                "NAME_CONTRACT_STATUS": ["Approved", "Refused"],
                "TARGET": [0, 1],
            }),
            "PREV_APPLICATION_CREDIT_RATIO",
        ),
        (
            "installment_aggregates",
            lambda: pd.DataFrame({
                "SK_ID_CURR": [1, 1],
                "SK_ID_PREV": [20, 20],
                "AMT_INSTALMENT": [100.0, 100.0],
                "AMT_PAYMENT": [100.0, 90.0],
                "DAYS_INSTALMENT": [10, 20],
                "DAYS_ENTRY_PAYMENT": [12, 18],
                "TARGET": [0, 1],
            }),
            "INSTALL_PAYMENT_RATIO_MEAN",
        ),
        (
            "pos_cash_aggregates",
            lambda: pd.DataFrame({
                "SK_ID_CURR": [1, 1],
                "SK_ID_PREV": [20, 20],
                "SK_DPD": [0, 2],
                "TARGET": [0, 1],
            }),
            "POS_SK_DPD_LATE_RATE",
        ),
        (
            "credit_card_aggregates",
            lambda: pd.DataFrame({
                "SK_ID_CURR": [1, 1],
                "SK_ID_PREV": [20, 20],
                "AMT_BALANCE": [50.0, 25.0],
                "AMT_CREDIT_LIMIT_ACTUAL": [100.0, 100.0],
                "TARGET": [0, 1],
            }),
            "CREDIT_CARD_UTILISATION_MEAN",
        ),
    ],
)
def test_aggregator_customer_index_contract(aggregator_name, frame, expected_column):
    import scripts.aggregate_external_data as aggregation

    result = getattr(aggregation, aggregator_name)(frame())
    assert result.index.name == "SK_ID_CURR"
    assert result.index.is_unique
    assert "TARGET" not in result.columns
    assert expected_column in result.columns


def test_data_run_smoke(tmp_path, monkeypatch):
    from src.data.run import main

    frame = pd.DataFrame({
        "SK_ID_CURR": range(20),
        "TARGET": [0, 1] * 10,
        "AMT_INCOME_TOTAL": [1000.0] * 20,
        "AMT_CREDIT": [2000.0] * 20,
        "AMT_ANNUITY": [500.0] * 20,
        "AMT_GOODS_PRICE": [1900.0] * 20,
        "DAYS_EMPLOYED": [-100] * 20,
        "CODE_GENDER": ["F", "M"] * 10,
        "NAME_CONTRACT_TYPE": ["Cash"] * 20,
    })
    source_dir = tmp_path / "data"
    source_dir.mkdir()
    frame.to_csv(source_dir / "application_train.csv", index=False)
    monkeypatch.chdir(tmp_path)
    main("data/application_train.csv", "data/processed")
    assert (source_dir / "processed" / "train.parquet").exists()
    assert (source_dir / "processed" / "test.parquet").exists()


def test_external_data_main_loads_installments(tmp_path):
    from scripts.aggregate_external_data import main

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({
        "SK_ID_PREV": [1],
        "SK_ID_CURR": [10],
        "NUM_INSTALMENT_VERSION": [1.0],
        "NUM_INSTALMENT_NUMBER": [1.0],
        "DAYS_INSTALMENT": [-10.0],
        "DAYS_ENTRY_PAYMENT": [-9.0],
        "AMT_INSTALMENT": [100.0],
        "AMT_PAYMENT": [100.0],
    }).to_csv(data_dir / "installments_payments.csv", index=False)

    main(data_dir, tmp_path / "processed")

    result = pd.read_parquet(tmp_path / "processed" / "installments_payments_agg.parquet")
    assert "INSTALL_AMT_PAYMENT_MEAN_AGG" in result.columns
