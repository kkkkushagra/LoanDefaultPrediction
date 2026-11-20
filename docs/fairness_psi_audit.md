# Task 7 Fairness and PSI Audit

Reproduce with: `.\.venv\Scripts\python.exe -m scripts.run_fairness_psi_audit`

Uses the saved fitted model, preprocessor, and feature pipeline. It does not train or refit model artifacts. Gender and age are joined from the audit frame by `SK_ID_CURR` and are used only to calculate reporting groups; they are not sent to the scorer. Review flags use the model artifact's current cutoff.

A group is labeled `small` when it has fewer than 100 applicants; its metrics are shown but should be interpreted cautiously. Missing groups are listed explicitly, with `n/a` metrics where no denominator exists.

- Held-out test applications: 61,503
- Positive cases: 4,965
- Current review threshold: 0.242685465

## Gender groups

| Group | N | Status | Observed default rate | Flag rate | Recall | False-positive rate |
|---|---:|---|---:|---:|---:|---:|
| F | 40,561 | ok | 6.99% | 2.22% | 11.07% | 1.55% |
| M | 20,940 | ok | 10.17% | 3.13% | 11.04% | 2.24% |
| Missing / unknown | 0 | missing | n/a | n/a | n/a | n/a |
| XNA | 2 | small | 0.00% | 0.00% | n/a | 0.00% |

## Age groups

| Group | N | Status | Observed default rate | Flag rate | Recall | False-positive rate |
|---|---:|---|---:|---:|---:|---:|
| 18-24 | 2,355 | ok | 11.80% | 3.86% | 10.07% | 3.03% |
| 25-34 | 14,304 | ok | 10.55% | 3.87% | 13.78% | 2.70% |
| 35-44 | 16,912 | ok | 8.24% | 2.65% | 10.98% | 1.90% |
| 45-54 | 14,041 | ok | 7.39% | 2.46% | 11.75% | 1.71% |
| 55-64 | 12,166 | ok | 5.61% | 0.95% | 5.57% | 0.68% |
| 65+ | 1,725 | ok | 3.71% | 0.06% | 0.00% | 0.06% |
| Under 18 | 0 | missing | n/a | n/a | n/a | n/a |
| Missing / unknown | 0 | missing | n/a | n/a | n/a | n/a |

## Population Stability Index

No comparable post-training scoring snapshot was available. The saved training split is the reference; the saved held-out test split is the baseline comparison. The drift demo is synthetic: test `AMT_INCOME_TOTAL` values are multiplied by 1.25; all other fields are unchanged. Numeric PSI uses ten reference quantile bins plus a separate missing bin; categorical PSI reports unseen comparison categories separately. A 0.5 count pseudocount avoids zero-frequency log terms. Interpretation bands are common heuristics, not project acceptance criteria. The Home Credit employed-day sentinel 365243 is treated as missing, matching scoring.

| Snapshot | Field | PSI | Heuristic interpretation |
|---|---|---:|---|
| Train vs held-out test baseline | AMT_CREDIT | 0.000109 | little change |
| Train vs held-out test baseline | AMT_INCOME_TOTAL | 0.000226 | little change |
| Train vs held-out test baseline | AMT_ANNUITY | 0.000249 | little change |
| Train vs held-out test baseline | AMT_GOODS_PRICE | 0.000123 | little change |
| Train vs held-out test baseline | DAYS_EMPLOYED | 0.000168 | little change |
| Train vs held-out test baseline | NAME_CONTRACT_TYPE | 0.000003 | little change |
| Train vs held-out test baseline | NAME_INCOME_TYPE | 0.000310 | little change |
| Simulated test shift (+25% income) | AMT_CREDIT | 0.000109 | little change |
| Simulated test shift (+25% income) | AMT_INCOME_TOTAL | 0.735613 | large change |
| Simulated test shift (+25% income) | AMT_ANNUITY | 0.000249 | little change |
| Simulated test shift (+25% income) | AMT_GOODS_PRICE | 0.000123 | little change |
| Simulated test shift (+25% income) | DAYS_EMPLOYED | 0.000168 | little change |
| Simulated test shift (+25% income) | NAME_CONTRACT_TYPE | 0.000003 | little change |
| Simulated test shift (+25% income) | NAME_INCOME_TYPE | 0.000310 | little change |

- Train-vs-test mean PSI: 0.000170
- Train-vs-test max PSI: 0.000310
- Simulated-shift mean PSI: 0.105225
- Simulated-shift max PSI: 0.735613

The simulated snapshot is a demonstration only and does not indicate observed production drift.
