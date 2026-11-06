import json
import argparse
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .training import (
    RANDOM_STATE,
    ks_statistic,
    load_model_data,
    metrics_at_threshold,
    model_factories,
    tune_lightgbm,
    tune_xgb,
)


def evaluate(y_true, probability, threshold=None):
    result = {
        "auc_roc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "ks": ks_statistic(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }
    result.update(metrics_at_threshold(y_true, probability, threshold))
    return result


def main(
    train_path="data/processed/train.parquet",
    test_path="data/processed/test.parquet",
    trials=30,
):
    optuna_available = trials > 0
    train_x, train_y, test_x, test_y = load_model_data(train_path, test_path)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scale_pos_weight = (len(train_y) - int(train_y.sum())) / int(train_y.sum())
    models = model_factories(scale_pos_weight)
    results = {}
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("loan-default-model-v1")

    tuned = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline_{name}"):
            oof_probability = cross_val_predict(
                model, train_x, train_y, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            cv_auc = roc_auc_score(train_y, oof_probability)
            mlflow.log_metric("cv_auc_roc_mean", float(cv_auc))
            mlflow.log_params(model.get_params())
            model.fit(train_x, train_y)
            probability = model.predict_proba(test_x)[:, 1]
            threshold = metrics_at_threshold(train_y, oof_probability)["threshold"]
            results[name] = evaluate(test_y, probability, threshold)
            results[name]["cv_auc_roc"] = cv_auc
            for metric, value in results[name].items():
                mlflow.log_metric(f"test_{metric}", float(value))
            mlflow.sklearn.log_model(model, name=name)

    if optuna_available:
        candidates = {}
        for name, tuner in (
            ("xgboost_tuned", tune_xgb),
            ("lightgbm_tuned", tune_lightgbm),
        ):
            model, cv_auc = tuner(train_x, train_y, trials, cv)
            candidates[name] = model
            with mlflow.start_run(run_name=name):
                mlflow.log_metric("cv_auc_roc_mean", cv_auc)
                mlflow.log_params(model.get_params())
                model.fit(train_x, train_y)
                probability = model.predict_proba(test_x)[:, 1]
                oof_probability = cross_val_predict(
                    model, train_x, train_y, cv=cv, method="predict_proba", n_jobs=-1
                )[:, 1]
                threshold = metrics_at_threshold(train_y, oof_probability)["threshold"]
                results[name] = evaluate(test_y, probability, threshold)
                results[name]["cv_auc_roc"] = cv_auc
                for metric, value in results[name].items():
                    mlflow.log_metric(f"test_{metric}", float(value))
                mlflow.sklearn.log_model(model, name=name)
                tuned[name] = model

    all_models = {**models, **tuned}
    best_name = max(results, key=lambda name: results[name]["cv_auc_roc"])
    best_model = all_models[best_name]
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"model": best_model, "feature_columns": train_x.columns.tolist()}, "models/model_v1.joblib")

    rows = [{"model": name, **values} for name, values in results.items()]
    pd.DataFrame(rows).sort_values("auc_roc", ascending=False).to_csv("docs/model_comparison.csv", index=False)
    comparison = pd.DataFrame(rows).sort_values("auc_roc", ascending=False)
    lines = [
        "# Model Comparison",
        "",
        "| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        cv_auc = row.get("cv_auc_roc", np.nan)
        cv_text = f"{cv_auc:.6f}" if pd.notna(cv_auc) else "-"
        lines.append(
            f"| {row['model']} | {cv_text} | {row['auc_roc']:.6f} | {row['pr_auc']:.6f} | {row['ks']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['brier']:.6f} | {row['threshold']:.6f} |"
        )
    selected = comparison.loc[comparison["model"] == best_name].iloc[0]
    auc_met = selected["auc_roc"] >= 0.78
    ks_met = selected["ks"] >= 0.35
    lines += [
        "",
        f"- **Best model:** `{best_name}`",
        f"- **AUC >= 0.78:** {'MET' if auc_met else 'NOT MET'} ({selected['auc_roc']:.6f})",
        f"- **KS >= 0.35:** {'MET' if ks_met else 'NOT MET'} ({selected['ks']:.6f})",
        "- The test set was used only for final evaluation. Thresholds were selected using train-only out-of-fold predictions.",
        "",
    ]
    Path("docs/model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return best_name, results[best_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    main(trials=args.trials)
