"""Score all processed reviews with the sentiment model and save the predictions,
so the dashboard, topics and API read them instead of running the model each time.

Usage:
    python -m src.pipeline.score
    python -m src.pipeline.score --model tfidf
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.predict import predict

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions.csv"


def score(df, model=None):
    labels, confidence, name = predict(df["text"], model)
    return pd.DataFrame({
        "reviewId": df["reviewId"].to_numpy(),
        "pred": labels,
        "confidence": confidence.round(3),
        "model": name,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xlmr", "tfidf"], default=None)
    args = parser.parse_args()

    df = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    start = time.perf_counter()
    predictions = score(df, args.model)
    seconds = time.perf_counter() - start

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    print(f"Scored {len(predictions)} reviews with {predictions['model'].iloc[0]} in {seconds:.0f} s")
    print(predictions["pred"].value_counts(normalize=True).round(3).to_string())
    print(f"Saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()