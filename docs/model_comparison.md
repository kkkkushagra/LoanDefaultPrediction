# Model Comparison

| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold | Recall at 70% Threshold | Precision at 70% Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blend_mean | 0.773255 | 0.776523 | 0.272939 | 0.416517 | 0.250810 | 0.451964 | 0.322599 | 0.179775 | 0.648253 | 0.699698 | 0.175328 |
| xgboost_tuned | 0.773872 | 0.776024 | 0.272537 | 0.416900 | 0.255136 | 0.447734 | 0.325048 | 0.172552 | 0.640909 | 0.700050 | 0.175200 |
| xgboost | 0.772193 | 0.775890 | 0.272760 | 0.416131 | 0.249343 | 0.458409 | 0.322997 | 0.175350 | 0.643046 | 0.700201 | 0.174376 |
| lightgbm_tuned | 0.773233 | 0.775443 | 0.272411 | 0.415627 | 0.268093 | 0.421551 | 0.327748 | 0.157974 | 0.627226 | 0.700050 | 0.175381 |
| lightgbm | 0.770259 | 0.774963 | 0.270322 | 0.416015 | 0.258662 | 0.430010 | 0.323020 | 0.173316 | 0.659136 | 0.700050 | 0.172809 |
| logistic_regression | 0.762382 | 0.764793 | 0.253400 | 0.401742 | 0.245083 | 0.419134 | 0.309304 | 0.196962 | 0.675520 | 0.700302 | 0.166121 |
| random_forest | 0.753009 | 0.756069 | 0.235273 | 0.389528 | 0.235199 | 0.412085 | 0.299473 | 0.068518 | 0.175781 | 0.698943 | 0.161702 |

- **Run mode:** `full`
- **Scale pos weight:** `enabled`
- Scale-pos-weight AUC/Brier comparison is reported by these modes when the run is repeated with and without `--no-scale-pos-weight`.
- **Best model:** `xgboost_tuned`
- **AUC >= 0.78:** NOT MET (0.776024)
- **KS >= 0.35:** MET (0.416900)
- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.
