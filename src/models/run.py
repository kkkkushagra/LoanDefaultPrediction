import json
import argparse
import time
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
import lightgbm as lgb


def evaluate(y_true, probability, threshold=None):
    result = {
        "auc_roc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "ks": ks_statistic(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
    }
    result.update(metrics_at_threshold(y_true, probability, threshold))
    return result


def metrics_at_recall(y_true, probability, recall_target=0.70):
    metrics = metrics_at_threshold(y_true, probability)
    thresholds = np.linspace(float(np.min(probability)), float(np.max(probability)), 1001)
    achievable = []
    for threshold in thresholds:
        prediction = (probability >= threshold).astype(int)
        tp = int(((prediction == 1) & (np.asarray(y_true) == 1)).sum())
        fn = int(((prediction == 0) & (np.asarray(y_true) == 1)).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        fp = int(((prediction == 1) & (np.asarray(y_true) == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        achievable.append((abs(recall - recall_target), threshold, recall, precision))
    _, threshold, recall, precision = min(achievable)
    return {
        "recall_70_threshold": threshold,
        "recall_at_70_threshold": recall,
        "precision_at_70_threshold": precision,
    }


def main(
    train_path="data/processed/train.parquet",
    test_path="data/processed/test.parquet",
    trials=30,
    fast=False,
    top_features=150,
):
    if top_features is not None and top_features < 1:
        raise ValueError("top_features must be positive or None")
    stage_start = time.perf_counter()
    optuna_available = trials > 0
    train_x, train_y, test_x, test_y = load_model_data(train_path, test_path)
    print(f"Data loading: {time.perf_counter() - stage_start:.2f}s")

    if top_features is not None:
        stage_start = time.perf_counter()
        selector = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        selector.fit(train_x, train_y)
        gain = selector.booster_.feature_importance(importance_type="gain")
        selected = [
            column
            for _, column in sorted(
                zip(gain, train_x.columns),
                key=lambda item: item[0],
                reverse=True,
            )[:top_features]
        ]
        train_x = train_x[selected]
        test_x = test_x[selected]
        print(f"Top-feature selection: {time.perf_counter() - stage_start:.2f}s")

    cv = StratifiedKFold(n_splits=3 if fast else 5, shuffle=True, random_state=RANDOM_STATE)
    scale_pos_weight = (len(train_y) - int(train_y.sum())) / int(train_y.sum())
    models = model_factories(scale_pos_weight)
    if fast:
        models = {name: model for name, model in models.items() if name != "random_forest"}
        for model in models.values():
            if model.__class__.__name__ == "XGBClassifier":
                model.set_params(tree_method="hist")
    results = {}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("loan-default-model-v1")

    tuned = {}
    stage_start = time.perf_counter()
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
            recall_metrics = metrics_at_recall(train_y, oof_probability)
            results[name].update(recall_metrics)
            for metric, value in results[name].items():
                mlflow.log_metric(f"test_{metric}", float(value))
            mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
    print(f"Baseline models: {time.perf_counter() - stage_start:.2f}s")

    if optuna_available:
        candidates = {}
        tune_x, tune_y = train_x, train_y
        if fast:
            tune_x, _, tune_y, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
                train_x,
                train_y,
                train_size=0.30,
                random_state=RANDOM_STATE,
                stratify=train_y,
            )
        stage_start = time.perf_counter()
        for name, tuner in (
            ("xgboost_tuned", tune_xgb),
            ("lightgbm_tuned", tune_lightgbm),
        ):
            tuned_trials = 10 if fast else trials
            model, cv_auc = tuner(tune_x, tune_y, tuned_trials, cv, fast=fast)
            model.fit(train_x, train_y)
            candidates[name] = model
            oof_probability = cross_val_predict(
                model, train_x, train_y, cv=cv, method="predict_proba", n_jobs=-1
            )[:, 1]
            cv_auc = roc_auc_score(train_y, oof_probability)
            with mlflow.start_run(run_name=name):
                mlflow.log_metric("cv_auc_roc_mean", cv_auc)
                mlflow.log_params(model.get_params())
                model.fit(train_x, train_y)
                probability = model.predict_proba(test_x)[:, 1]
                threshold = metrics_at_threshold(train_y, oof_probability)["threshold"]
                results[name] = evaluate(test_y, probability, threshold)
                results[name]["cv_auc_roc"] = cv_auc
                results[name].update(metrics_at_recall(train_y, oof_probability))
                for metric, value in results[name].items():
                    mlflow.log_metric(f"test_{metric}", float(value))
                mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
                tuned[name] = model
        print(f"Hyperparameter tuning and refit: {time.perf_counter() - stage_start:.2f}s")

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
        "| Model | CV AUC-ROC | Test AUC-ROC | PR-AUC | KS | Precision | Recall | F1 | Brier | Threshold | Recall at 70% Threshold | Precision at 70% Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparison.iterrows():
        cv_auc = row.get("cv_auc_roc", np.nan)
        cv_text = f"{cv_auc:.6f}" if pd.notna(cv_auc) else "-"
        lines.append(
            f"| {row['model']} | {cv_text} | {row['auc_roc']:.6f} | {row['pr_auc']:.6f} | {row['ks']:.6f} | "
            f"{row['precision']:.6f} | {row['recall']:.6f} | {row['f1']:.6f} | {row['brier']:.6f} | {row['threshold']:.6f} | "
            f"{row.get('recall_at_70_threshold', np.nan):.6f} | {row.get('precision_at_70_threshold', np.nan):.6f} |"
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
    print(f"Model comparison export: {time.perf_counter() - stage_start:.2f}s")
    return best_name, results[best_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--fast", action="store_true", help="Use the reduced tuning workload")
    parser.add_argument("--top-features", type=int, default=150, metavar="N")
    args = parser.parse_args()
    main(trials=args.trials, fast=args.fast, top_features=args.top_features)
