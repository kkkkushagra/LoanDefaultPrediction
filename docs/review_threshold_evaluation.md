# Validation-Selected Review Threshold

Reproduce with: `.\.venv\Scripts\python.exe -m scripts.evaluate_review_threshold --target-recall 0.70`

The evaluation loads the saved `model_v2.joblib`, preprocessor, and feature pipeline. It does not train or refit any model or transformer.

The validation fold is the 20% validation portion of the saved training split, reconstructed using the calibration workflow's stratified 40% holdout split, 50% validation split, and random seed 42. Raw seven-field application records are matched by `SK_ID_CURR`. The test split is used only for the final comparison.

The selected threshold is the available validation-score cutoff whose recall is closest to 70%; ties use the highest cutoff.

## Validation selection

- Applications: 49,202
- Positive cases: 3,972
- Selected threshold: 0.103250489
- Achieved validation recall: 69.813696%

## Held-out test comparison

- Applications: 61,503
- Positive cases: 4,965

| Threshold | Recall | Precision | Flagged for review |
|---|---:|---:|---:|
| Current (0.242685465) | 11.278953% | 34.398034% | 2.647025% (1,628) |
| Validation-selected (0.103250489) | 70.231621% | 14.480296% | 39.154188% (24,081) |

The dashboard's threshold was not changed.
