import numpy as np

from src.sentiment.threshold import apply_threshold


def test_low_confidence_becomes_neutral():
    proba = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
    pred = apply_threshold(proba, ["negative", "positive"], threshold=0.7)
    assert pred.tolist() == ["negative", "neutral", "positive"]


def test_threshold_of_half_keeps_everything():
    proba = np.array([[0.51, 0.49], [0.3, 0.7]])
    pred = apply_threshold(proba, ["negative", "positive"], threshold=0.5)
    assert pred.tolist() == ["negative", "positive"]