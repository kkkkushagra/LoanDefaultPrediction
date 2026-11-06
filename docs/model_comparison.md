# Model Comparison

| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.751932 | 0.758409 | 0.246554 | 0.384162 | 0.254565 | 0.387513 | 0.307275 | 0.185548 | 0.672367 |
| xgboost_tuned | 0.754138 | 0.758018 | 0.245913 | 0.386603 | 0.244839 | 0.408459 | 0.306159 | 0.236961 | 0.727005 |
| lightgbm | 0.750323 | 0.756751 | 0.243418 | 0.384299 | 0.225787 | 0.446526 | 0.299919 | 0.182693 | 0.631957 |
| lightgbm_tuned | 0.732905 | 0.756652 | 0.244324 | 0.382140 | 0.233466 | 0.437261 | 0.304403 | 0.191365 | 0.650927 |
| logistic_regression | 0.748387 | 0.751438 | 0.235947 | 0.371293 | 0.227461 | 0.432427 | 0.298112 | 0.201782 | 0.654821 |
| random_forest | 0.736302 | 0.741188 | 0.222083 | 0.356679 | 0.229004 | 0.395972 | 0.290185 | 0.070943 | 0.213010 |

- **Best model:** `xgboost_tuned`
- **AUC >= 0.78:** NOT MET (0.758018)
- **KS >= 0.35:** MET (0.386603)
- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.
