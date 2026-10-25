"""TF-IDF + LinearSVC baseline for sentiment, trained on star-based labels.

Usage:
    python -m src.sentiment.baseline
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.evaluation.gold import load_gold
from src.preprocessing.prepare import PROCESSED_PATH

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "baseline_tfidf_svm.joblib"
METRICS_PATH = ROOT / "reports" / "metrics" / "baseline.json"
FIGURE_PATH = ROOT / "reports" / "figures" / "baseline_confusion_gold.png"
LABELS = ["negative", "neutral", "positive"]


def build_model():
    features = FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])
    return Pipeline([
        ("tfidf", features),
        ("svm", LinearSVC(C=0.5, class_weight="balanced")),
    ])


def evaluate(name, model, texts, labels):
    start = time.perf_counter()
    pred = model.predict(texts)
    seconds_per_1000 = (time.perf_counter() - start) / len(texts) * 1000

    print(f"\n{name} (n={len(labels)})")
    print(classification_report(labels, pred, labels=LABELS, digits=3, zero_division=0))
    report = classification_report(
        labels, pred, labels=LABELS, zero_division=0, output_dict=True
    )
    result = {
        "n": len(labels),
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "precision": round(report["macro avg"]["precision"], 3),
        "recall": round(report["macro avg"]["recall"], 3),
        "f1_per_class": {label: round(report[label]["f1-score"], 3) for label in LABELS},
        "inference_s_per_1000": round(seconds_per_1000, 4),
    }
    return pred, result


def f1_by_lang(df, pred):
    scores = {}
    for lang in ["uk", "ru", "en", "mixed"]:
        mask = (df["lang"] == lang).to_numpy()
        if mask.sum() > 0:
            scores[lang] = round(f1_score(df["sentiment"][mask], pred[mask], average="macro"), 3)
    return scores


def main():
    df = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    gold = load_gold().dropna(subset=["sentiment"])
    train = df[(df["split"] == "train") & ~df["reviewId"].isin(gold["reviewId"])]
    val = df[df["split"] == "val"]

    model = build_model()
    start = time.perf_counter()
    model.fit(train["text"], train["sentiment"])
    train_time = time.perf_counter() - start
    print(f"Trained on {len(train)} reviews in {train_time:.1f} s")

    val_pred, val_metrics = evaluate("Validation, star labels", model, val["text"], val["sentiment"])
    gold_pred, gold_metrics = evaluate("Manual labels", model, gold["text"], gold["sentiment"])
    val_metrics["macro_f1_by_lang"] = f1_by_lang(val, val_pred)
    print("Validation macro-F1 by language:", val_metrics["macro_f1_by_lang"])

    metrics = {
        "model": "tfidf_words_chars + LinearSVC",
        "train_size": len(train),
        "train_time_s": round(train_time, 1),
        "val_stars": val_metrics,
        "gold_manual": gold_metrics,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(gold["sentiment"], gold_pred, labels=LABELS, cmap="Blues")
    plt.title("TF-IDF + SVM, manual labels")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved metrics to {METRICS_PATH}, model to {MODEL_PATH}")


if __name__ == "__main__":
    main()