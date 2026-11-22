import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

from backend.service import ScoringService


RANDOM_STATE = 42
TARGET_RECALL = 0.70
APPLICATION_FIELDS = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def validation_positions(target: pd.Series) -> np.ndarray:
    _, holdout = train_test_split(
        np.arange(len(target)),
        test_size=0.40,
        stratify=target,
        random_state=RANDOM_STATE,
    )
    _, validation = train_test_split(
        holdout,
        test_size=0.50,
        stratify=target.iloc[holdout],
        random_state=RANDOM_STATE,
    )
    return validation


def select_review_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    recall_target: float = TARGET_RECALL,
) -> tuple[float, float]:
    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must be in (0, 1]")
    _, recall, thresholds = precision_recall_curve(target, probability)
    selected = min(
        range(len(thresholds)),
        key=lambda index: (
            abs(float(recall[index]) - recall_target),
            -float(thresholds[index]),
        ),
    )
    return float(thresholds[selected]), float(recall[selected])


def threshold_metrics(target: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    target = np.asarray(target, dtype=int)
    flagged = np.asarray(probability) >= threshold
    true_positive = int(np.sum(flagged & (target == 1)))
    false_positive = int(np.sum(flagged & (target == 0)))
    positive_count = int(np.sum(target == 1))
    return {
        "threshold": float(threshold),
        "recall": true_positive / positive_count if positive_count else 0.0,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        ),
        "flagged_count": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
    }


def load_applications(membership: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    applications = membership[["SK_ID_CURR"]].merge(
        raw,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
    )
    if applications["AMT_CREDIT"].isna().any():
        raise ValueError("Some split IDs did not match raw application records")
    return applications


def score_applications(service: ScoringService, applications: pd.DataFrame) -> np.ndarray:
    transformed = service.transform(applications)
    return service.predict_calibrated(transformed)


def write_report(
    path: Path,
    command: str,
    validation: dict,
    test: dict,
    target_recall: float,
) -> None:
    lines = [
        "# Validation-Selected Review Threshold",
        "",
        f"Reproduce with: `{command}`",
        "",
        "The evaluation loads the saved `model_v2.joblib`, preprocessor, and feature pipeline. "
        "It does not train or refit any model or transformer.",
        "",
        f"The validation fold is the 20% validation portion of the saved training split, "
        f"reconstructed using the calibration workflow's stratified 40% holdout split, "
        f"50% validation split, and random seed {RANDOM_STATE}. Raw seven-field application "
        "records are matched by `SK_ID_CURR`. The test split is used only for the final "
        "comparison.",
        "",
        f"The selected threshold is the available validation-score cutoff whose recall is "
        f"closest to {target_recall:.0%}; ties use the highest cutoff.",
        "",
        "## Validation selection",
        "",
        f"- Applications: {validation['sample_size']:,}",
        f"- Positive cases: {validation['positive_count']:,}",
        f"- Selected threshold: {validation['selected_threshold']:.9f}",
        f"- Achieved validation recall: {validation['selected_recall']:.6%}",
        "",
        "## Held-out test comparison",
        "",
        f"- Applications: {test['sample_size']:,}",
        f"- Positive cases: {test['positive_count']:,}",
        "",
        "| Threshold | Recall | Precision | Flagged for review |",
        "|---|---:|---:|---:|",
        (
            f"| Current ({test['current']['threshold']:.9f}) "
            f"| {test['current']['recall']:.6%} "
            f"| {test['current']['precision']:.6%} "
            f"| {test['current']['flagged_share']:.6%} "
            f"({test['current']['flagged_count']:,}) |"
        ),
        (
            f"| Validation-selected ({test['selected']['threshold']:.9f}) "
            f"| {test['selected']['recall']:.6%} "
            f"| {test['selected']['precision']:.6%} "
            f"| {test['selected']['flagged_share']:.6%} "
            f"({test['selected']['flagged_count']:,}) |"
        ),
        "",
        "The Current row uses the dashboard's validation-curve default (ASSUMED FN=5, FP=1); the target-recall candidate is not active.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(
    train_path: Path,
    test_path: Path,
    raw_path: Path,
    target_recall: float,
    output_json: Path,
    output_report: Path,
) -> dict:
    if not 0.0 < target_recall <= 1.0:
        raise ValueError("target_recall must be in (0, 1]")

    train_membership = pd.read_parquet(train_path, columns=["SK_ID_CURR", "TARGET"])
    test_membership = pd.read_parquet(test_path, columns=["SK_ID_CURR", "TARGET"])
    validation_indices = validation_positions(train_membership["TARGET"])
    validation_membership = train_membership.iloc[validation_indices].reset_index(drop=True)
    test_membership = test_membership.reset_index(drop=True)

    raw_applications = pd.read_csv(
        raw_path,
        usecols=["SK_ID_CURR", *APPLICATION_FIELDS],
    )
    validation_applications = load_applications(validation_membership, raw_applications)
    test_applications = load_applications(test_membership, raw_applications)
    service = ScoringService(PROJECT_ROOT)
    validation_probability = score_applications(service, validation_applications)
    validation_target = validation_membership["TARGET"].to_numpy(dtype=int)
    selected_threshold, selected_recall = select_review_threshold(
        validation_target,
        validation_probability,
        target_recall,
    )
    test_probability = score_applications(service, test_applications)
    test_target = test_membership["TARGET"].to_numpy(dtype=int)
    current_threshold = service.cost_threshold
    result = {
        "model": "models/model_v2.joblib",
        "input_fields": APPLICATION_FIELDS,
        "target_validation_recall": target_recall,
        "threshold_selection": "closest achievable validation recall; highest cutoff breaks ties",
        "validation": {
            "split": "calibration validation fold reconstructed from train.parquet",
            "random_state": RANDOM_STATE,
            "sample_size": len(validation_target),
            "positive_count": int(validation_target.sum()),
            "current_threshold": threshold_metrics(
                validation_target,
                validation_probability,
                current_threshold,
            ),
            "selected_threshold": selected_threshold,
            "selected_recall": selected_recall,
        },
        "test": {
            "split": str(test_path),
            "sample_size": len(test_target),
            "positive_count": int(test_target.sum()),
            "current": threshold_metrics(
                test_target,
                test_probability,
                current_threshold,
            ),
            "selected": threshold_metrics(
                test_target,
                test_probability,
                selected_threshold,
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    command = (
        ".\\.venv\\Scripts\\python.exe -m scripts.evaluate_review_threshold "
        f"--target-recall {target_recall:.2f}"
    )
    write_report(output_report, command, result["validation"], result["test"], target_recall)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select and evaluate a review threshold for seven-field scoring")
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/processed/train.parquet")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data/processed/test.parquet")
    parser.add_argument("--raw-path", type=Path, default=PROJECT_ROOT / "data/application_train.csv")
    parser.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "docs/review_threshold_evaluation.json",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=PROJECT_ROOT / "docs/review_threshold_evaluation.md",
    )
    arguments = parser.parse_args()
    main(
        train_path=arguments.train_path,
        test_path=arguments.test_path,
        raw_path=arguments.raw_path,
        target_recall=arguments.target_recall,
        output_json=arguments.output_json,
        output_report=arguments.output_report,
    )
