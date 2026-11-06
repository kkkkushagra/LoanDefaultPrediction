# Model Comparison

| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.751932 | 0.758409 | 0.246554 | 0.384162 | 0.254565 | 0.387513 | 0.307275 | 0.185548 | 0.672367 |
| xgboost_tuned | 0.753985 | 0.758177 | 0.245341 | 0.384833 | 0.230657 | 0.446123 | 0.304091 | 0.130515 | 0.512762 |
| lightgbm | 0.750323 | 0.756751 | 0.243418 | 0.384299 | 0.225787 | 0.446526 | 0.299919 | 0.182693 | 0.631957 |
| lightgbm_tuned | 0.733454 | 0.755374 | 0.245420 | 0.380420 | 0.240419 | 0.411883 | 0.303615 | 0.119063 | 0.495014 |
| logistic_regression | 0.748387 | 0.751438 | 0.235947 | 0.371293 | 0.227461 | 0.432427 | 0.298112 | 0.201782 | 0.654821 |
| random_forest | 0.736302 | 0.741188 | 0.222083 | 0.356679 | 0.229004 | 0.395972 | 0.290185 | 0.070943 | 0.213010 |

- **Best model:** `xgboost`
- **AUC >= 0.78:** NOT MET (0.758409)
- **KS >= 0.35:** MET (0.384162)
- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.
