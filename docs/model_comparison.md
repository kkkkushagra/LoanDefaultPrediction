# Model Comparison

| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold | Recall at 70% Threshold | Precision at 70% Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| blend_mean | 0.782097 | 0.784328 | 0.282406 | 0.430376 | 0.264410 | 0.461027 | 0.336074 | 0.174686 | 0.652736 | 0.699446 | 0.182390 |
| xgboost_tuned | 0.781684 | 0.783885 | 0.283456 | 0.430934 | 0.273627 | 0.445519 | 0.339030 | 0.163326 | 0.643218 | 0.700302 | 0.182558 |
| lightgbm_tuned | 0.781102 | 0.783184 | 0.282300 | 0.427235 | 0.270091 | 0.450151 | 0.337613 | 0.115868 | 0.514667 | 0.700554 | 0.181998 |
| xgboost | 0.779933 | 0.782986 | 0.281149 | 0.428995 | 0.269268 | 0.443303 | 0.335033 | 0.170160 | 0.659002 | 0.700403 | 0.180854 |
| lightgbm | 0.778399 | 0.781085 | 0.279132 | 0.425139 | 0.261589 | 0.461430 | 0.333892 | 0.167963 | 0.648525 | 0.699799 | 0.178029 |
| logistic_regression | 0.773189 | 0.775238 | 0.262664 | 0.414527 | 0.254847 | 0.442095 | 0.323317 | 0.192369 | 0.675703 | 0.699698 | 0.175568 |
| random_forest | 0.759470 | 0.764771 | 0.240595 | 0.403723 | 0.233255 | 0.462236 | 0.310051 | 0.068014 | 0.163532 | 0.701057 | 0.165900 |

- **Run mode:** `full`
- **Scale pos weight:** `enabled`
- Scale-pos-weight AUC/Brier comparison is reported by these modes when the run is repeated with and without `--no-scale-pos-weight`.
- **Best model:** `blend_mean`
- **AUC >= 0.78:** MET (0.784328)
- **KS >= 0.35:** MET (0.430376)
- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.
