"""Neutral class via confidence threshold: a negative/positive classifier,
low-confidence predictions are labeled neutral. Threshold is tuned on half
of the manual labels (dev) and evaluated on the other half (test).

Usage:
    python -m src.sentiment.threshold
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation.gold import load_gold
from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.baseline import LABELS, MODEL_PATH as BASELINE_PATH, build_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "threshold_lr.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "threshold.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "threshold_curve.png"
THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.01), 2)


def build_binary_model():
    features = build_model().named_steps["tfidf"]
    return Pipeline([
        ("tfidf", features),
        ("lr", LogisticRegression(C=4, max_iter=2000, class_weight="balanced")),
    ])


def apply_threshold(proba, classes, threshold):
    pred = np.array(classes, dtype=object)[proba.argmax(axis=1)]
    pred[proba.max(axis=1) < threshold] = "neutral"
    return pred


def tune_threshold(proba, classes, labels):
    scores = [f1_score(labels, apply_threshold(proba, classes, t), average="macro") for t in THRESHOLDS]
    return THRESHOLDS[int(np.argmax(scores))], scores


def summary(labels, pred):
    report = classification_report(labels, pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "f1_per_class": {label: round(report[label]["f1-score"], 3) for label in LABELS},
    }


def main():
    df = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    gold = load_gold().dropna(subset=["sentiment"])
    dev, test = train_test_split(gold, test_size=0.5, stratify=gold["sentiment"], random_state=42)

    train = df[(df["split"] == "train") & ~df["reviewId"].isin(gold["reviewId"])]
    train = train[train["sentiment"] != "neutral"]

    model = build_binary_model()
    start = time.perf_counter()
    model.fit(train["text"], train["sentiment"])
    train_time = time.perf_counter() - start
    classes = list(model.classes_)
    print(f"Trained on {len(train)} reviews without 3-star ones in {train_time:.1f} s")

    threshold, dev_scores = tune_threshold(model.predict_proba(dev["text"]), classes, dev["sentiment"])
    print(f"Best threshold on dev (n={len(dev)}): {threshold}, macro-F1 {max(dev_scores):.3f}")

    start = time.perf_counter()
    test_pred = apply_threshold(model.predict_proba(test["text"]), classes, threshold)
    seconds_per_1000 = (time.perf_counter() - start) / len(test) * 1000
    print(f"\nThreshold model, test half of manual labels (n={len(test)})")
    print(classification_report(test["sentiment"], test_pred, labels=LABELS, digits=3, zero_division=0))

    baseline_pred = joblib.load(BASELINE_PATH).predict(test["text"])
    print("Baseline 3-class SVM, same test half")
    print(classification_report(test["sentiment"], baseline_pred, labels=LABELS, digits=3, zero_division=0))

    metrics = {
        "model": "tfidf_words_chars + LogisticRegression (neg/pos) + threshold",
        "train_size": len(train),
        "train_time_s": round(train_time, 1),
        "threshold": float(threshold),
        "dev_n": len(dev),
        "test_n": len(test),
        "threshold_model": summary(test["sentiment"], test_pred),
        "baseline_svm": summary(test["sentiment"], baseline_pred),
        "inference_s_per_1000": round(seconds_per_1000, 4),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(THRESHOLDS, dev_scores, marker=".")
    plt.axvline(threshold, color="gray", linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("Macro-F1 (dev)")
    plt.title("Neutral by confidence threshold")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold)}, MODEL_PATH)
    print(f"Saved metrics to {METRICS_PATH}, model to {MODEL_PATH}")


if __name__ == "__main__":
    main()