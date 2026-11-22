import math
import json
from pathlib import Path

import numpy as np

DEFAULT_FN_COST = 5.0
DEFAULT_FP_COST = 1.0


def load_validation_curve(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Stored validation threshold curve not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_validation_curve(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 1000,
) -> dict:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if bins < 1:
        raise ValueError("bins must be positive")
    if target.ndim != 1 or probability.ndim != 1 or len(target) != len(probability):
        raise ValueError("target and probability must be one-dimensional arrays of equal length")
    if len(target) == 0 or not np.isin(target, [0, 1]).all():
        raise ValueError("target must contain binary labels")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probability values must be finite and in [0, 1]")

    score_bins = np.floor(probability * bins).astype(int).clip(0, bins)
    grouped = []
    for score_bin in np.unique(score_bins):
        selected = score_bins == score_bin
        positive = int(target[selected].sum())
        total = int(selected.sum())
        grouped.append({
            "bin": int(score_bin),
            "positive": positive,
            "negative": total - positive,
        })

    return {
        "split": "calibration_validation",
        "score_bins": bins,
        "sample_size": int(len(target)),
        "positive_count": int(target.sum()),
        "groups": grouped,
    }


def analyze_validation_curve(
    curve: dict,
    fn_cost: float,
    fp_cost: float,
) -> dict:
    if not math.isfinite(fn_cost) or not math.isfinite(fp_cost) or fn_cost <= 0 or fp_cost <= 0:
        raise ValueError("costs must be finite and positive")

    bins = int(curve["score_bins"])
    sample_size = int(curve["sample_size"])
    positive_count = int(curve["positive_count"])
    if bins < 1 or sample_size < 1 or not 0 <= positive_count <= sample_size:
        raise ValueError("validation curve metadata is invalid")

    positive = np.zeros(bins + 1, dtype=np.int64)
    negative = np.zeros(bins + 1, dtype=np.int64)
    for group in curve["groups"]:
        score_bin = int(group["bin"])
        group_positive = int(group["positive"])
        group_negative = int(group["negative"])
        if not 0 <= score_bin <= bins or group_positive < 0 or group_negative < 0:
            raise ValueError("validation curve contains an invalid score group")
        positive[score_bin] += group_positive
        negative[score_bin] += group_negative
    if int(positive.sum() + negative.sum()) != sample_size:
        raise ValueError("validation curve counts do not match sample_size")
    if int(positive.sum()) != positive_count:
        raise ValueError("validation curve counts do not match positive_count")

    flagged_positive = np.cumsum(positive[::-1])[::-1]
    flagged_negative = np.cumsum(negative[::-1])[::-1]
    candidates = []
    for score_bin in range(bins + 1):
        true_positive = int(flagged_positive[score_bin])
        false_positive = int(flagged_negative[score_bin])
        false_negative = positive_count - true_positive
        cost = false_negative * fn_cost + false_positive * fp_cost
        candidates.append((cost, score_bin, true_positive, false_positive))

    _, best_bin, true_positive, false_positive = min(
        candidates,
        key=lambda item: (item[0], -item[1]),
    )
    flagged_count = true_positive + false_positive
    return {
        "threshold": best_bin / bins,
        "flagged_percent": flagged_count / sample_size * 100,
        "recall": true_positive / positive_count if positive_count else 0.0,
        "precision": true_positive / flagged_count if flagged_count else 0.0,
        "validation_sample_size": sample_size,
        "assumed_cost_warning": "ASSUMED costs; not observed lender costs.",
    }
