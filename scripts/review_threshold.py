import math

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.service import SENSITIVE_FIELDS, ScoringService
from src.models.training import RANDOM_STATE, load_model_features

ID = "SK_ID_CURR"
TARGET = "TARGET"
FIELDS = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
]
RECALL_TARGET = 0.70
BATCH_SIZE = 10_000


def score_batch(service, batch):
    # Match the seven-field form. The applicant ID is used only for joining labels.
    columns = [
        name for name in service.preprocessor.input_columns_
        if name not in SENSITIVE_FIELDS
    ]
    application = batch.reindex(columns=columns)
    application["DAYS_EMPLOYED_ANOMALY"] = 0

    processed = service.preprocessor.transform(application)
    # Preserve the preprocessor's single-row treatment of numeric 0 and 1.
    for name in FIELDS:
        if name in service.preprocessor.numeric_columns_:
            is_flag_value = application[name].isin([0, 1])
            processed.loc[is_flag_value, name] = application.loc[is_flag_value, name]

    engineered = service.feature_pipeline.transform(application.assign(SK_ID_CURR=-1))
    matrix = pd.concat([processed, engineered], axis=1)
    return service.predict_calibrated(load_model_features(matrix))


def score_split(service, raw, split, name):
    cases = raw.loc[split[ID].tolist()]
    labels = split[TARGET].to_numpy()
    if not np.array_equal(cases[TARGET].to_numpy(), labels):
        raise ValueError(f"{name} labels do not match the raw CSV.")

    inputs = cases[FIELDS].reset_index(drop=True)
    # The form leaves employment duration blank when it is unknown.
    inputs["DAYS_EMPLOYED"] = inputs["DAYS_EMPLOYED"].replace(365243, np.nan)

    print(f"Scoring {len(inputs):,} {name} applications...", flush=True)
    probabilities = np.concatenate([
        score_batch(service, inputs.iloc[start:start + BATCH_SIZE])
        for start in range(0, len(inputs), BATCH_SIZE)
    ])

    # Check the batch calculation against the API's single-applicant scorer.
    for index in range(min(5, len(inputs))):
        payload = inputs.iloc[index].dropna().to_dict()
        np.testing.assert_allclose(
            probabilities[index], service.predict(payload), atol=1e-7, rtol=0,
        )

    return labels, probabilities


def print_result(name, threshold, labels, probabilities):
    flagged = probabilities >= threshold
    defaulted = labels == 1
    caught = int((flagged & defaulted).sum())
    flagged_count = int(flagged.sum())
    defaults = int(defaulted.sum())
    precision = caught / flagged_count if flagged_count else 0.0
    print(
        f"{name}: cutoff={threshold:.6%} | "
        f"recall={caught / defaults:.2%} ({caught:,}/{defaults:,}) | "
        f"precision={precision:.2%} | "
        f"flagged={flagged.mean():.2%} ({flagged_count:,}/{len(labels):,})"
    )


def main():
    raw = pd.read_csv(
        "data/application_train.csv", usecols=[ID, TARGET, *FIELDS],
    ).set_index(ID, verify_integrity=True)
    train = pd.read_parquet("data/processed/train.parquet", columns=[ID, TARGET])
    test = pd.read_parquet("data/processed/test.parquet", columns=[ID, TARGET])

    if not train[ID].is_unique or not test[ID].is_unique or train[ID].isin(test[ID]).any():
        raise ValueError("Expected unique, disjoint train and test applicant IDs.")

    # Recreate the validation split used by src/models/calibration.py.
    labels = train[TARGET]
    _, holdout = train_test_split(
        np.arange(len(train)), test_size=0.40,
        stratify=labels, random_state=RANDOM_STATE,
    )
    _, validation = train_test_split(
        holdout, test_size=0.50,
        stratify=labels.iloc[holdout], random_state=RANDOM_STATE,
    )

    service = ScoringService()
    val_y, val_p = score_split(service, raw, train.iloc[validation], "validation")
    positive_scores = np.sort(val_p[val_y == 1])[::-1]
    rank = math.ceil(RECALL_TARGET * len(positive_scores))
    candidate = float(positive_scores[rank - 1])
    print_result("Validation candidate", candidate, val_y, val_p)

    test_y, test_p = score_split(service, raw, test, "test")
    print_result("Test at current cutoff", service.cost_threshold, test_y, test_p)
    print_result("Test at candidate cutoff", candidate, test_y, test_p)


if __name__ == "__main__":
    main()