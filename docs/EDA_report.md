# EDA Report

## Dataset

Real Home Credit file analyzed: `data/application_train.csv`.
Rows: 307,511; columns: 122.

## Validation and cleaning

- Required columns: `SK_ID_CURR` and `TARGET`.
- `SK_ID_CURR` must be unique and non-negative.
- `TARGET` must be binary (`0` or `1`).
- Duplicate applicant records are removed.
- Infinite values are converted to missing values.
- Sensitive and potentially identifying attributes are excluded before modeling.

## Train-only preprocessing

Numeric missing values and outlier bounds are learned only from training data. Categorical missing values use training modes. Categorical values are one-hot encoded with an unknown-category-safe encoder.

## Real findings

### Class balance

- Non-default (`TARGET=0`): 282,686 rows (91.9271%).
- Default (`TARGET=1`): 24,825 rows (8.0729%).
- Imbalance ratio: 11.39 non-default rows per default row.

### Top missing columns

| Column | Missing |
|---|---:|
| COMMONAREA_AVG | 69.87% |
| COMMONAREA_MODE | 69.87% |
| COMMONAREA_MEDI | 69.87% |
| NONLIVINGAPARTMENTS_MEDI | 69.43% |
| NONLIVINGAPARTMENTS_MODE | 69.43% |
| NONLIVINGAPARTMENTS_AVG | 69.43% |
| FONDKAPREMONT_MODE | 68.39% |
| LIVINGAPARTMENTS_AVG/MEDI/MODE | 68.35% |
| FLOORSMIN_AVG/MODE/MEDI | 67.85% |

### Default rate by segment

| Segment | Rows | Default rate |
|---|---:|---:|
| Female applicants | 202,448 | 7.00% |
| Male applicants | 105,059 | 10.14% |
| Lower secondary education | 3,816 | 10.93% |
| Secondary education | 218,391 | 8.94% |
| Incomplete higher education | 10,277 | 8.48% |
| Higher education | 74,863 | 5.36% |
| Academic degree | 164 | 1.83% |

`DAYS_EMPLOYED=365243` occurs in 55,374 rows (18.0072%). These sentinels are converted to nulls and represented by `DAYS_EMPLOYED_ANOMALY`.
