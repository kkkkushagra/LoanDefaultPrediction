from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from .training import load_model_features


def human_feature_name(column: str) -> str:
    label = column
    for prefix in ("WOE_", "RAW_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    return label.replace("_", " ").title()


def prediction_shap_values(explainer, matrix: pd.DataFrame, row: int = 0):
    values = explainer.shap_values(matrix.iloc[[row]])
    if isinstance(values, list):
        return np.asarray(values[1 if len(values) > 1 else 0][0])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[0, :, 1] if values.shape[2] > 1 else values[0, :, 0]
    return values[0]


def explain_applicant(
    base_model,
    matrix: pd.DataFrame,
    row: int = 0,
    top_n: int = 5,
) -> dict:
    explainer = shap.TreeExplainer(base_model)
    contributions = prediction_shap_values(explainer, matrix, row)
    ranking = np.argsort(contributions)
    raising = ranking[-top_n:][::-1]
    lowering = ranking[:top_n]
    return {
        "association_note": "Model associations, not causes.",
        "raising_risk": [
            {"feature": human_feature_name(matrix.columns[index]), "shap_value": float(contributions[index])}
            for index in raising
        ],
        "lowering_risk": [
            {"feature": human_feature_name(matrix.columns[index]), "shap_value": float(contributions[index])}
            for index in lowering
        ],
    }


def calculate_global_importance(
    base_model,
    matrix: pd.DataFrame,
    sample_size: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    sample = matrix.sample(n=min(sample_size, len(matrix)), random_state=seed)
    explainer = shap.TreeExplainer(base_model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):
        values = values[1 if len(values) > 1 else 0]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[2] > 1 else values[:, :, 0]
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({
            "feature": sample.columns,
            "human_feature": [human_feature_name(column) for column in sample.columns],
            "mean_absolute_shap": importance,
        })
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )


def main(
    test_path="data/processed/test.parquet",
    model_path="models/model_v2.joblib",
    output_path="docs/shap_global.csv",
):
    model = joblib.load(model_path)
    frame = pd.read_parquet(test_path)
    feature_columns = model["features"]
    normalized = load_model_features(frame.drop(columns=["TARGET"], errors="ignore"))
    matrix = normalized.reindex(columns=feature_columns).fillna(0.0)
    importance = calculate_global_importance(model["base_model"], matrix)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    print(f"Saved {len(importance)} global SHAP features to {output_path}")


if __name__ == "__main__":
    main()
