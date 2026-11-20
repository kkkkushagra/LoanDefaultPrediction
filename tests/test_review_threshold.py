import numpy as np
import pytest

from scripts.evaluate_review_threshold import select_review_threshold


def test_select_review_threshold_uses_closest_recall_and_highest_cutoff_on_tie():
    target = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.8, 0.1])

    threshold, recall = select_review_threshold(target, probability, recall_target=0.5)

    assert threshold == 0.9
    assert recall == 0.5


def test_select_review_threshold_rejects_invalid_target():
    target = np.array([1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.8, 0.1])

    with pytest.raises(ValueError, match="recall_target must be in"):
        select_review_threshold(target, probability, recall_target=1.1)
