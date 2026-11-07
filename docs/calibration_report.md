# Calibration Report

## Selected model and calibration

- Base model: retained/logged `xgboost_tuned` parameters.
- Calibration selection: validation-fold Brier score.
- Selected method: `isotonic`.

| Method | Validation Brier |
|---|---:|
| isotonic | 0.066666 |
| sigmoid | 0.066671 |

## Test metrics

| Metric | Before calibration | After calibration |
|---|---:|---:|
| BRIER | 0.159433 | 0.067716 |
| LOG_LOSS | 0.486885 | 0.246611 |
| ECE | 0.271649 | 0.027994 |
| AUC | 0.781905 | 0.781905 |
| KS | 0.432412 | 0.432412 |

- Bootstrap 95% CI for test AUC (calibrated): [0.775026, 0.788417]
- AUC and KS invariance check: passed; calibration preserves ranking.

## Reliability curves

| Before mean probability | Before observed rate | After mean probability | After observed rate |
|---:|---:|---:|---:|
| 0.009917086652576818 | 0.07086859915088763 | 0.009917086652576818 | 0.013058071597967456 |
| 0.01902439024390244 | 0.12636301843616052 | 0.01902439024390244 | 0.02612374779272045 |
| 0.022926829268292682 | 0.17358311192533835 | 0.022926829268292682 | 0.03129034814595432 |
| 0.02959349593495935 | 0.2236466531491861 | 0.02959349593495935 | 0.038305731161063755 |
| 0.042919850430824254 | 0.2784003375667108 | 0.042919850430824254 | 0.06014480964628386 |
| 0.04959349593495935 | 0.34130625729153796 | 0.04959349593495935 | 0.07246889347191504 |
| 0.07284552845528455 | 0.41531877397521727 | 0.07284552845528455 | 0.1010415807778467 |
| 0.10829268292682927 | 0.5041107090023474 | 0.10829268292682927 | 0.14427609647642808 |
| 0.1510569105691057 | 0.6141746036696241 | 0.1510569105691057 | 0.20532480710327236 |
| 0.30108925377987317 | 0.7759824744803482 | 0.30108925377987317 | 0.39515838333472597 |

## Risk bands

Thresholds selected only from validation data: Low < 0.070406; Medium < 0.167995; High >= 0.167995.

| Band | Applicant share | Observed default rate |
|---|---:|---:|
| Low | 56.000520% | 2.766971% |
| Medium | 23.998829% | 8.340108% |
| High | 20.000650% | 22.607918% |

## Cost-based threshold

> ASSUMED placeholders, not real bank costs.

- Assumed false-negative cost (`--cost-fn`): 5
- Assumed false-positive cost (`--cost-fp`): 1
- Validation-selected threshold: 0.242685
- Expected normalized test cost at selected threshold: 0.320846
- Expected normalized test cost at 0.5: 0.363251

## Limitations

- Prototype only; suitability for production has not been established.
- Costs are assumed placeholders and are not real bank costs.
- No causal claims are made.
- Results depend on a single train/validation/test split and limited bootstrap resampling.
