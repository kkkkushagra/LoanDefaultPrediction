import json
from pathlib import Path
import re

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
EXCLUDED_COLUMNS = {"SK_ID_CURR", "TARGET"}
SENSITIVE_COLUMNS = {
    "CODE_GENDER",
    "DAYS_BIRTH",
    "AGE_YEARS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
}


def load_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.drop(columns=sorted(EXCLUDED_COLUMNS | SENSITIVE_COLUMNS), errors="ignore").fillna(0.0)
    names = {}
    used = set()
    for column in features.columns:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", str(column))
        if not safe or safe[0].isdigit():
            safe = f"f_{safe}"
        base = safe
        suffix = 2
        while safe in used:
            safe = f"{base}_{suffix}"
            suffix += 1
        used.add(safe)
        names[column] = safe
    return features.rename(columns=names)


def load_model_data(train_path="data/processed/train.parquet", test_path="data/processed/test.parquet"):
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)
    if "TARGET" not in train or "TARGET" not in test:
        raise ValueError("TARGET is required in train and test")
    train_x = load_model_features(train)
    test_x = load_model_features(test)
    train_y = train["TARGET"].astype(int)
    test_y = test["TARGET"].astype(int)
    if train_x.columns.tolist() != test_x.columns.tolist():
        raise ValueError("Train and test model columns differ")
    return train_x, train_y, test_x, test_y


def ks_statistic(y_true, probability):
    positives = probability[np.asarray(y_true) == 1]
    negatives = probability[np.asarray(y_true) == 0]
    thresholds = np.linspace(0, 1, 1001)
    positive_cdf = np.array([(positives <= threshold).mean() for threshold in thresholds])
    negative_cdf = np.array([(negatives <= threshold).mean() for threshold in thresholds])
    return float(np.max(np.abs(positive_cdf - negative_cdf)))


def metrics_at_threshold(y_true, probability, threshold=None):
    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_true, probability)
        f1 = np.divide(
            2 * precision[:-1] * recall[:-1],
            precision[:-1] + recall[:-1],
            out=np.zeros_like(precision[:-1]),
            where=(precision[:-1] + recall[:-1]) != 0,
        )
        threshold = float(thresholds[int(np.argmax(f1))])
    prediction = (probability >= threshold).astype(int)
    tp = int(((prediction == 1) & (np.asarray(y_true) == 1)).sum())
    fp = int(((prediction == 1) & (np.asarray(y_true) == 0)).sum())
    fn = int(((prediction == 0) & (np.asarray(y_true) == 1)).sum())
    precision_score = tp / (tp + fp) if tp + fp else 0.0
    recall_score = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision_score * recall_score / (precision_score + recall_score) if precision_score + recall_score else 0.0
    return {
        "threshold": threshold,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }


def model_factories(scale_pos_weight):
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=RANDOM_STATE,
        )),
    ])
    return {
        "logistic_regression": logistic,
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            metric="auc",
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
    }


def tune_xgb(train_x, train_y, trials, cv, fast=False):
    positive = int(train_y.sum())
    negative = len(train_y) - positive
    trial_iterations = {}

    def objective(trial):
        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 800),
            learning_rate=0.1 if fast else trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", negative / positive * 0.5, negative / positive * 1.5),
            eval_metric="aucpr",
            early_stopping_rounds=50,
            tree_method="hist" if fast else None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        scores = []
        iterations = []
        for fit_index, valid_index in cv.split(train_x, train_y):
            fit_x, valid_x = train_x.iloc[fit_index], train_x.iloc[valid_index]
            fit_y, valid_y = train_y.iloc[fit_index], train_y.iloc[valid_index]
            model.fit(fit_x, fit_y, eval_set=[(valid_x, valid_y)], verbose=False)
            scores.append(roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1]))
            iterations.append(int(model.best_iteration or model.n_estimators))
        trial_iterations[len(trial_iterations)] = int(np.median(iterations))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = study.best_params.copy()
    params.update(
        n_estimators=max(1, trial_iterations[study.best_trial.number]),
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return xgb.XGBClassifier(**params), study.best_value


def tune_lightgbm(train_x, train_y, trials, cv, fast=False):
    positive = int(train_y.sum())
    negative = len(train_y) - positive
    trial_iterations = {}

    def objective(trial):
        model = lgb.LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 250, 800),
            learning_rate=0.1 if fast else trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            metric="auc",
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            max_depth=trial.suggest_int("max_depth", -1, 10),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", negative / positive * 0.5, negative / positive * 1.5),
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        scores = []
        iterations = []
        for fit_index, valid_index in cv.split(train_x, train_y):
            fit_x, valid_x = train_x.iloc[fit_index], train_x.iloc[valid_index]
            fit_y, valid_y = train_y.iloc[fit_index], train_y.iloc[valid_index]
            model.fit(
                fit_x, fit_y,
                eval_set=[(valid_x, valid_y)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            scores.append(roc_auc_score(valid_y, model.predict_proba(valid_x)[:, 1]))
            iterations.append(int(model.best_iteration_))
        trial_iterations[len(trial_iterations)] = int(np.median(iterations))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    params = study.best_params.copy()
    params.update(
        n_estimators=max(1, trial_iterations[study.best_trial.number]),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    params["metric"] = "auc"
    return lgb.LGBMClassifier(**params), study.best_value
