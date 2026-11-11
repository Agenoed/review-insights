"""Compare training labels for the sentiment model: stars, LLM labels and LLM labels
with synthetic neutral reviews. All variants are evaluated on the test half of the
manual labels. Also checks how the model works on a category it has not seen.

Usage:
    python -m src.sentiment.experiments
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation.gold import load_gold
from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.baseline import LABELS, MODEL_PATH as BASELINE_PATH, build_model
from src.sentiment.threshold import MODEL_PATH as STARS_THRESHOLD_PATH, apply_threshold

ROOT = Path(__file__).resolve().parents[2]
LLM_LABELS_PATH = ROOT / "data" / "llm" / "labels_train_openrouter.csv"
SYNTHETIC_PATH = ROOT / "data" / "synthetic" / "neutral.csv"
MODEL_PATH = ROOT / "models" / "llm_synthetic_lr.joblib"
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
THRESHOLDS = np.round(np.arange(0.35, 0.96, 0.01), 2)

CONTROL = [
    "Додайте будь ласка темну тему",
    "Як змінити номер телефону в застосунку?",
    "Після оновлення не проходить оплата",
    "Дуже зручний додаток, дякую",
]


def build_lr():
    return Pipeline([
        ("tfidf", build_model().named_steps["tfidf"]),
        ("lr", LogisticRegression(C=4, max_iter=2000, class_weight="balanced")),
    ])


def summary(labels, pred):
    report = classification_report(labels, pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "macro_f1": round(report["macro avg"]["f1-score"], 3),
        "accuracy": round(accuracy_score(labels, pred), 3),
        "f1_per_class": {label: round(report[label]["f1-score"], 3) for label in LABELS},
    }


def tune_threshold(model, dev):
    proba, classes = model.predict_proba(dev["text"]), list(model.classes_)
    scores = [f1_score(dev["sentiment"], apply_threshold(proba, classes, t), average="macro") for t in THRESHOLDS]
    return float(THRESHOLDS[int(np.argmax(scores))])


def load_data():
    reviews = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    gold = load_gold().dropna(subset=["sentiment"])
    dev, test = train_test_split(gold, test_size=0.5, stratify=gold["sentiment"], random_state=42)

    stars = reviews[(reviews["split"] == "train") & ~reviews["reviewId"].isin(gold["reviewId"])]
    llm = pd.read_csv(LLM_LABELS_PATH).merge(
        reviews[["reviewId", "text", "category", "lang", "sentiment"]].rename(columns={"sentiment": "stars"}),
        on="reviewId",
    )
    llm = llm[~llm["reviewId"].isin(gold["reviewId"])]
    synthetic = pd.read_csv(SYNTHETIC_PATH, keep_default_na=False).drop_duplicates("text")
    synthetic = synthetic[synthetic["llm_check"] == "neutral"]
    return dev, test, stars, llm, synthetic


def with_synthetic(texts, labels, synthetic):
    return (
        pd.concat([texts, synthetic["text"]], ignore_index=True),
        pd.concat([labels, pd.Series(["neutral"] * len(synthetic))], ignore_index=True),
    )


def compare_labels(dev, test, stars, llm, synthetic):
    train_sets = {
        "stars, all train": (stars["text"], stars["sentiment"]),
        "stars, same 5k reviews": (llm["text"], llm["stars"]),
        "LLM labels, 5k": (llm["text"], llm["llm_sentiment"]),
        "LLM labels + synthetic": with_synthetic(llm["text"], llm["llm_sentiment"], synthetic),
    }
    results, predictions, models = {}, {}, {}
    for name, (texts, labels) in train_sets.items():
        model = build_lr().fit(texts, labels)
        models[name] = model
        results[name] = summary(test["sentiment"], model.predict(test["text"]))
        predictions[name] = model.predict(CONTROL)
        print(f"{name:<28} macro-F1 {results[name]['macro_f1']}  neutral F1 {results[name]['f1_per_class']['neutral']}")

    best = models["LLM labels + synthetic"]
    threshold = tune_threshold(best, dev)
    name = "LLM labels + synthetic + threshold"
    results[name] = summary(test["sentiment"], apply_threshold(best.predict_proba(test["text"]), list(best.classes_), threshold))
    results[name]["threshold"] = threshold
    predictions[name] = apply_threshold(best.predict_proba(CONTROL), list(best.classes_), threshold)
    print(f"{name:<28} macro-F1 {results[name]['macro_f1']}  neutral F1 {results[name]['f1_per_class']['neutral']}  (threshold {threshold})")

    baseline = joblib.load(BASELINE_PATH)
    results["reference: SVM on stars"] = summary(test["sentiment"], baseline.predict(test["text"]))
    saved = joblib.load(STARS_THRESHOLD_PATH)
    stars_threshold_pred = apply_threshold(saved["model"].predict_proba(test["text"]), list(saved["model"].classes_), saved["threshold"])
    results["reference: stars + threshold"] = summary(test["sentiment"], stars_threshold_pred)

    print("\nControl examples:")
    print(pd.DataFrame(predictions, index=CONTROL).T.to_string())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best, "threshold": threshold}, MODEL_PATH)
    return results


def cross_category(llm, synthetic):
    results = {}
    print("\nUnseen category (macro-F1 on LLM labels of that category):")
    for category in sorted(llm["category"].unique()):
        own = llm[llm["category"] == category]
        own_train, own_test = train_test_split(own, test_size=0.5, random_state=42)
        others = llm[llm["category"] != category]
        other_synthetic = synthetic[synthetic["category"] != category]
        scores = {}
        for setup, train_df in {"seen": pd.concat([others, own_train]), "unseen": others}.items():
            texts, labels = with_synthetic(train_df["text"], train_df["llm_sentiment"], other_synthetic)
            pred = build_lr().fit(texts, labels).predict(own_test["text"])
            scores[setup] = round(f1_score(own_test["llm_sentiment"], pred, average="macro"), 3)
        results[category] = {"n_test": len(own_test), **scores}
        print(f"{category:<15} n={len(own_test):<4} seen {scores['seen']}  unseen {scores['unseen']}")
    return results


def plot(results, cross):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    names = list(results)
    x = np.arange(len(names))
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - 0.2, [results[n]["macro_f1"] for n in names], 0.4, label="macro-F1")
    plt.bar(x + 0.2, [results[n]["f1_per_class"]["neutral"] for n in names], 0.4, label="F1 neutral")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.title("Training labels, test half of manual labels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiments_labels.png", dpi=150)

    categories = list(cross)
    x = np.arange(len(categories))
    plt.figure(figsize=(10, 4.5))
    plt.bar(x - 0.2, [cross[c]["seen"] for c in categories], 0.4, label="category seen in training")
    plt.bar(x + 0.2, [cross[c]["unseen"] for c in categories], 0.4, label="category not seen")
    plt.xticks(x, categories, rotation=25, ha="right")
    plt.ylabel("macro-F1")
    plt.title("Model on a new category")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "experiments_categories.png", dpi=150)


def main():
    dev, test, stars, llm, synthetic = load_data()
    print(f"Manual labels: dev {len(dev)}, test {len(test)}. LLM-labeled: {len(llm)}. Synthetic neutral: {len(synthetic)}\n")
    results = compare_labels(dev, test, stars, llm, synthetic)
    cross = cross_category(llm, synthetic)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "experiments.json").write_text(
        json.dumps({"labels": results, "categories": cross}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot(results, cross)
    print(f"\nSaved metrics to {METRICS_DIR / 'experiments.json'}, figures to {FIGURES_DIR}, model to {MODEL_PATH}")


if __name__ == "__main__":
    main()