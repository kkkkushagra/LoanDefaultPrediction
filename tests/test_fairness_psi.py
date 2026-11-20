import numpy as np
import pandas as pd
import pytest

from scripts.run_fairness_psi_audit import (
    group_metrics,
    join_applications,
    population_stability_index,
)


def test_psi_is_zero_for_identical_numeric_distribution():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])

    assert population_stability_index(values, values.copy(), numeric=True) == pytest.approx(0.0)


def test_psi_detects_numeric_and_categorical_distribution_shifts():
    reference = pd.Series([1, 1, 1, 2, 2, 2])
    numeric_shift = pd.Series([10, 10, 10, 20, 20, 20])
    categorical_shift = pd.Series(["a", "a", "a", "b", "b", "b"])
    categorical_comparison = pd.Series(["new"] * 6)

    assert population_stability_index(reference, numeric_shift, numeric=True) > 0
    assert population_stability_index(
        categorical_shift,
        categorical_comparison,
        numeric=False,
    ) > 0


def test_group_metrics_reports_missing_small_and_zero_positive_groups():
    population = pd.DataFrame({
        "gender": ["F", "F", "M", "M"],
        "TARGET": [1, 0, 0, 0],
        "probability": [0.8, 0.2, 0.1, 0.3],
    })

    result = group_metrics(
        population,
        "gender",
        ["F", "M", "Missing / unknown", "Small"],
        threshold=0.25,
        minimum_group_size=3,
    )

    assert result[0]["status"] == "small"
    assert result[0]["recall"] == 1.0
    assert result[1]["recall"] is None
    assert result[1]["false_positive_rate"] == pytest.approx(1 / 2)
    assert result[2]["status"] == "missing"
    assert result[2]["observed_default_rate"] is None


def test_join_applications_treats_employment_sentinel_as_missing():
    split = pd.DataFrame({"SK_ID_CURR": [1], "TARGET": [0]})
    raw = pd.DataFrame({
        "SK_ID_CURR": [1],
        "AMT_CREDIT": [100.0],
        "AMT_INCOME_TOTAL": [50.0],
        "AMT_ANNUITY": [10.0],
        "AMT_GOODS_PRICE": [100.0],
        "DAYS_EMPLOYED": [365243],
        "NAME_CONTRACT_TYPE": ["Cash loans"],
        "NAME_INCOME_TYPE": ["Working"],
    })

    joined = join_applications(split, raw)

    assert pd.isna(joined.loc[0, "DAYS_EMPLOYED"])
