import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.service import ScoringService
from backend.thresholds import build_validation_curve
from src.models.training import RANDOM_STATE, load_model_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def create_curve(
    train_path: Path = PROJECT_ROOT / "data/processed/train.parquet",
    output_path: Path = PROJECT_ROOT / "docs/validation_threshold_curve.json",
    bins: int = 1000,
) -> dict:
    training = pd.read_parquet(train_path)
    labels = training["TARGET"].astype(int)
    _, holdout = train_test_split(
        np.arange(len(training)),
        test_size=0.40,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    _, validation = train_test_split(
        holdout,
        test_size=0.50,
        stratify=labels.iloc[holdout],
        random_state=RANDOM_STATE,
    )
    validation_frame = training.iloc[validation]

    service = ScoringService(PROJECT_ROOT)
    matrix = load_model_features(
        validation_frame.drop(columns=["TARGET"], errors="ignore")
    ).reindex(columns=service.model_feature_names).fillna(0.0)
    probability = service.predict_calibrated(matrix)
    curve = build_validation_curve(
        validation_frame["TARGET"].to_numpy(dtype=int),
        probability,
        bins=bins,
    )
    curve["random_state"] = RANDOM_STATE
    curve["quantization"] = "Scores rounded down into 0.001-wide bins; no applicant IDs stored."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(curve, separators=(",", ":")) + "\n", encoding="utf-8")
    return curve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build an aggregate threshold curve from the saved model's validation predictions"
    )
    parser.add_argument("--train-path", type=Path, default=PROJECT_ROOT / "data/processed/train.parquet")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "docs/validation_threshold_curve.json",
    )
    parser.add_argument("--bins", type=int, default=1000)
    args = parser.parse_args()
    result = create_curve(args.train_path, args.output_path, args.bins)
    print(json.dumps({key: value for key, value in result.items() if key != "groups"}, indent=2))
