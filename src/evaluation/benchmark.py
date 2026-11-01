"""Speed check of the main pipeline steps on the processed dataset.

Usage:
    python -m src.evaluation.benchmark
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing.language import detect_lang
from src.preprocessing.prepare import PROCESSED_PATH, clean_text
from src.sentiment.baseline import MODEL_PATH as BASELINE_PATH
from src.sentiment.threshold import MODEL_PATH as THRESHOLD_PATH, apply_threshold

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "reports" / "metrics" / "benchmark.json"


def seconds_per_1000(func, items):
    start = time.perf_counter()
    func(items)
    return round((time.perf_counter() - start) / len(items) * 1000, 4)


def main():
    texts = pd.read_csv(PROCESSED_PATH, keep_default_na=False)["text"]
    baseline = joblib.load(BASELINE_PATH)
    saved = joblib.load(THRESHOLD_PATH)
    model, threshold = saved["model"], saved["threshold"]

    def predict_threshold(items):
        return apply_threshold(model.predict_proba(items), list(model.classes_), threshold)

    results = {
        "clean_text": seconds_per_1000(lambda items: items.map(clean_text), texts),
        "detect_lang": seconds_per_1000(lambda items: items.map(detect_lang), texts.head(5000)),
        "baseline_predict": seconds_per_1000(baseline.predict, texts),
        "threshold_predict": seconds_per_1000(predict_threshold, texts),
    }

    print(f"Seconds per 1000 reviews (dataset: {len(texts)} reviews)")
    for key, value in results.items():
        print(f"{key:<20}{value}")
    results["n_reviews"] = len(texts)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()