"""Pretrained multilingual sentiment model without fine-tuning
(cardiffnlp/twitter-xlm-roberta-base-sentiment), evaluated on the manual labels
and compared with the best TF-IDF model on the same test half.

Usage:
    python -m src.sentiment.zero_shot
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline

from src.evaluation.gold import load_gold
from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.baseline import LABELS
from src.sentiment.experiments import CONTROL, MODEL_PATH as BEST_PATH, summary
from src.sentiment.threshold import apply_threshold

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
METRICS_PATH = ROOT / "reports" / "metrics" / "zero_shot.json"


def predict(classifier, texts, batch_size=64):
    output = classifier(list(texts), batch_size=batch_size, truncation=True, max_length=128)
    return [item["label"].lower() for item in output]


def f1_by_lang(df, pred):
    scores = {}
    for lang in ["uk", "ru", "en"]:
        mask = (df["lang"] == lang).to_numpy()
        if mask.any():
            scores[lang] = round(f1_score(df["sentiment"][mask], pred[mask], average="macro"), 3)
    return scores


def main():
    gold = load_gold().dropna(subset=["sentiment"])
    _, test = train_test_split(gold, test_size=0.5, stratify=gold["sentiment"], random_state=42)
    texts = pd.read_csv(PROCESSED_PATH, keep_default_na=False)["text"].head(5000)

    device = 0 if torch.cuda.is_available() else -1
    print("Device:", torch.cuda.get_device_name(0) if device == 0 else "CPU")
    classifier = pipeline("text-classification", model=MODEL_NAME, device=device)
    predict(classifier, texts.head(64))

    zero_shot_pred = pd.Series(predict(classifier, test["text"])).to_numpy()
    print(f"\nZero-shot XLM-T, test half of manual labels (n={len(test)})")
    print(classification_report(test["sentiment"], zero_shot_pred, labels=LABELS, digits=3, zero_division=0))

    saved = joblib.load(BEST_PATH)
    best = saved["model"]
    best_pred = apply_threshold(best.predict_proba(test["text"]), list(best.classes_), saved["threshold"])

    start = time.perf_counter()
    predict(classifier, texts)
    seconds_per_1000 = (time.perf_counter() - start) / len(texts) * 1000

    metrics = {
        "model": MODEL_NAME,
        "device": "cuda" if device == 0 else "cpu",
        "test_n": len(test),
        "zero_shot": {**summary(test["sentiment"], zero_shot_pred), "macro_f1_by_lang": f1_by_lang(test, zero_shot_pred)},
        "best_tfidf_llm_synthetic": {**summary(test["sentiment"], best_pred), "macro_f1_by_lang": f1_by_lang(test, best_pred)},
        "inference_s_per_1000": round(seconds_per_1000, 4),
    }
    print(f"zero-shot XLM-T          macro-F1 {metrics['zero_shot']['macro_f1']}  by language {metrics['zero_shot']['macro_f1_by_lang']}")
    print(f"TF-IDF, LLM + synthetic  macro-F1 {metrics['best_tfidf_llm_synthetic']['macro_f1']}  by language {metrics['best_tfidf_llm_synthetic']['macro_f1_by_lang']}")
    print(f"Inference: {seconds_per_1000:.3f} s per 1000 reviews")

    print("\nControl examples:")
    for text, label in zip(CONTROL, predict(classifier, CONTROL)):
        print(f"  {label:<9} {text}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()