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

## Real-data status

`data/application_train.csv` is not present in the repository, so notebook execution and numeric EDA findings cannot be reproduced in this checkout. Before submission, place the Home Credit training file in `data/`, run `python -m src.data.run`, and rerun the EDA notebook.

The report must not invent findings for a dataset that was not executed. Once real data is available, record the row count, target prevalence, missing-value leaders, class imbalance, main numeric distributions, categorical cardinality, sentinel rates, and train/test drift.
