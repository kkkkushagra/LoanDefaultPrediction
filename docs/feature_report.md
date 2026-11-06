# Feature Engineering Report

The fitted Home Credit feature pipeline used 246,008 training rows and evaluated the
engineered ratio, external-source, and available application-level candidates. Bureau
aggregates were not evaluated because `data/bureau.csv` is not present.

## Information Value

| Feature | IV | Retained (0.02 <= IV <= 0.50) |
|---|---:|---|
| EXT_SOURCE_MEAN | 0.609399 | Review (> 0.50); retained |
| EXT_SOURCE_MIN | 0.465855 | Yes |
| EXT_SOURCE_MAX | 0.443708 | Yes |
| DAYS_EMPLOYED | 0.111373 | Yes |
| CREDIT_TO_GOODS | 0.068926 | Yes |
| DEBT_TO_INCOME | 0.011111 | No |
| ANNUITY_TO_INCOME | 0.006953 | No |

## Applied Engineering

- Debt-to-income, annuity-to-income, and credit-to-goods ratios use zero-safe division.
- External source features are summarized as mean, minimum, and maximum.
- Missing and infinite values are replaced with medians, then zero, in transformed output.
- WoE maps and feature selection are fit only on training data.

## Final Matrix

The final real-data matrices contain 302 columns, including `SK_ID_CURR`, `TARGET`,
and no duplicate columns. Bureau aggregates are out of scope because
`data/bureau.csv` is not available in this checkout.

## Verification

- 19 pipeline tests pass.
- Source package coverage: 79%.
- Real train/test WoE zero-share is 0% for every retained WoE column.
