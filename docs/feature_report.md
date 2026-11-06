# Feature Engineering Report

The fitted Home Credit feature pipeline used 246,008 training rows and evaluated the
engineered ratio, external-source, and available application-level candidates. Bureau
aggregates were not evaluated because `data/bureau.csv` is not present.

## Information Value

| Feature | IV | Retained (0.02 <= IV <= 0.50) |
|---|---:|---|
| EXT_SOURCE_MEAN | 0.609767 | No |
| EXT_SOURCE_MIN | 0.466192 | Yes |
| EXT_SOURCE_MAX | 0.443916 | Yes |
| DAYS_EMPLOYED | 0.092081 | Yes |
| CREDIT_TO_GOODS | 0.072546 | Yes |
| EMPLOYED_TO_AGE | 0.067575 | Yes |
| DEBT_TO_INCOME | 0.011170 | No |
| ANNUITY_TO_INCOME | 0.006246 | No |

## Applied Engineering

- Debt-to-income, annuity-to-income, and credit-to-goods ratios use zero-safe division.
- Employed-to-age uses audit age supplied separately and is not emitted from the model frame.
- External source features are summarized as mean, minimum, and maximum.
- Missing and infinite values are replaced with medians, then zero, in transformed output.
- WoE maps and feature selection are fit only on training data.
