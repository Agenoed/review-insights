"""All sentiment models on the test half of the manual labels in one place:
Precision, Recall and F1 for every class, macro averages, macro-F1 by language,
confusion matrices, 95% bootstrap intervals and a paired bootstrap for the gap
between two models (both models are scored on the same resampled reviews).

Usage:
    python -m src.evaluation.sentiment_report
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.sentiment.baseline import LABELS, MODEL_PATH as BASELINE_PATH
from src.sentiment.experiments import MODEL_PATH as TFIDF_PATH, load_data
from src.sentiment.finetune_xlmr import MODELS_DIR, predict as predict_transformer
from src.sentiment.threshold import MODEL_PATH as THRESHOLD_PATH, apply_threshold

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "metrics" / "sentiment_report.json"
ZERO_SHOT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
SEED = 42
N_BOOT = 1000
PAIRS = [
    ("xlmr_finetuned", "tfidf_llm_synthetic"),
    ("xlmr_finetuned", "svm_stars"),
    ("xlmr_finetuned", "xlmt_finetuned"),
    ("tfidf_llm_synthetic", "svm_stars"),
]


def tfidf_pred(path, texts):
    saved = joblib.load(path)
    if isinstance(saved, dict):
        model = saved["model"]
        return apply_threshold(model.predict_proba(texts), list(model.classes_), saved["threshold"])
    return saved.predict(texts)


def transformer_pred(name, texts, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = MODELS_DIR / name
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
    return predict_transformer(model, AutoTokenizer.from_pretrained(path), list(texts), device)


def zero_shot_pred(texts, device):
    from transformers import pipeline

    classifier = pipeline("text-classification", model=ZERO_SHOT, device=0 if device.type == "cuda" else -1)
    output = classifier(list(texts), batch_size=64, truncation=True, max_length=128)
    return np.array([item["label"].lower() for item in output], dtype=object)


def macro_f1(y, pred):
    return f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)


def neutral_f1(y, pred):
    return f1_score(y, pred, labels=["neutral"], average="macro", zero_division=0)


def scores(y, pred, langs):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average="macro", zero_division=0)
    return {
        "macro_precision": round(float(mp), 3),
        "macro_recall": round(float(mr), 3),
        "macro_f1": round(float(mf), 3),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "per_class": {label: {"precision": round(float(p[i]), 3), "recall": round(float(r[i]), 3),
                              "f1": round(float(f[i]), 3), "support": int(s[i])} for i, label in enumerate(LABELS)},
        "macro_f1_by_lang": {lang: round(float(macro_f1(y[langs == lang], pred[langs == lang])), 3)
                             for lang in ["uk", "ru", "en"] if (langs == lang).any()},
        "confusion": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def interval(values):
    return [round(float(np.percentile(values, 2.5)), 3), round(float(np.percentile(values, 97.5)), 3)]


def bootstrap(y, pred, draws):
    return {
        "macro_f1": interval([macro_f1(y[i], pred[i]) for i in draws]),
        "neutral_f1": interval([neutral_f1(y[i], pred[i]) for i in draws]),
    }


def paired(y, a, b, draws):
    gaps = np.array([macro_f1(y[i], a[i]) - macro_f1(y[i], b[i]) for i in draws])
    return {
        "gap": round(float(macro_f1(y, a) - macro_f1(y, b)), 3),
        "ci95": interval(gaps),
        "share_not_better": round(float((gaps <= 0).mean()), 3),
    }


def main():
    _, test, _, _, _ = load_data()
    texts = test["text"].tolist()
    y = test["sentiment"].to_numpy()
    langs = test["lang"].to_numpy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Test half of the manual labels: {len(test)} reviews, {pd.Series(y).value_counts().to_dict()}")

    preds = {
        "svm_stars": tfidf_pred(BASELINE_PATH, texts),
        "threshold_stars": tfidf_pred(THRESHOLD_PATH, texts),
        "tfidf_llm_synthetic": tfidf_pred(TFIDF_PATH, texts),
        "xlmt_zero_shot": zero_shot_pred(texts, device),
        "xlmt_finetuned": transformer_pred("xlmt_finetuned", texts, device),
        "xlmr_finetuned": transformer_pred("xlmr_finetuned", texts, device),
    }
    preds = {name: np.asarray(pred, dtype=object) for name, pred in preds.items()}

    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

    results = {"test_n": len(y), "bootstrap": N_BOOT, "models": {}, "pairs": {}}
    for name, pred in preds.items():
        results["models"][name] = {**scores(y, pred, langs), "ci95": bootstrap(y, pred, draws)}
        m = results["models"][name]
        n = m["per_class"]["neutral"]
        print(f"{name:<22} macro P {m['macro_precision']}  R {m['macro_recall']}  F1 {m['macro_f1']} {m['ci95']['macro_f1']}  "
              f"neutral P {n['precision']} R {n['recall']} F1 {n['f1']} {m['ci95']['neutral_f1']}")

    print()
    for a, b in PAIRS:
        results["pairs"][f"{a} - {b}"] = paired(y, preds[a], preds[b], draws)
        print(f"{a} - {b}: {results['pairs'][f'{a} - {b}']}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {METRICS_PATH}")


if __name__ == "__main__":
    main()