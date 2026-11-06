# EDA Report

## Dataset

Place the Home Credit `application_train.csv` file in `data/`; large raw data must remain untracked.

## Validation and cleaning

- Required columns: `SK_ID_CURR` and `TARGET`.
- `SK_ID_CURR` must be unique and non-negative.
- `TARGET` must be binary (`0` or `1`).
- Duplicate applicant records are removed.
- Infinite values are converted to missing values.
- Sensitive and potentially identifying attributes are excluded before modeling.

## Train-only preprocessing

Numeric missing values and outlier bounds are learned only from training data. Categorical missing values use training modes. Categorical values are one-hot encoded with an unknown-category-safe encoder.

## Planned checks

Update this report after running the notebook with the real dataset:

- target prevalence and class imbalance
- missing-value rates by feature
- numeric distributions and trained outlier bounds
- categorical cardinality and rare-category effects
- train/test feature distributions and data drift
