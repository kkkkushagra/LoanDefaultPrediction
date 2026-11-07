# Model Comparison

| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold | Recall at 70% Threshold | Precision at 70% Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xgboost | 0.770363 | 0.775890 | 0.272760 | 0.416131 | 0.252044 | 0.447130 | 0.322370 | 0.175350 | 0.649668 | 0.700453 | 0.172345 |
| lightgbm | 0.767861 | 0.774963 | 0.270322 | 0.416015 | 0.251734 | 0.445921 | 0.321802 | 0.173316 | 0.648663 | 0.700453 | 0.170828 |
| xgboost_tuned | 0.757714 | 0.765506 | 0.260023 | 0.401400 | 0.240241 | 0.473515 | 0.318758 | 0.161626 | 0.608836 | 0.700050 | 0.163116 |
| logistic_regression | 0.762169 | 0.764793 | 0.253400 | 0.401742 | 0.243231 | 0.425176 | 0.309440 | 0.196962 | 0.672443 | 0.700252 | 0.165927 |
| lightgbm_tuned | 0.724163 | 0.723150 | 0.190351 | 0.338433 | 0.196997 | 0.459819 | 0.275825 | 0.072880 | 0.146796 | 0.706193 | 0.141541 |

- **Best model:** `xgboost`
- **AUC >= 0.78:** NOT MET (0.775890)
- **KS >= 0.35:** MET (0.416131)
- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.
