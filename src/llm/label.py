"""Sentiment labeling with an LLM (OpenRouter or local Ollama), in batches.
For the manual sample it also compares LLM labels and stars with manual labels.

Usage:
    python -m src.llm.label --source gold
    python -m src.llm.label --source train --n 3000
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

from src.evaluation.gold import load_gold
from src.llm.client import chat, get_model, get_provider
from src.preprocessing.prepare import PROCESSED_PATH, SENTIMENT

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "llm"
METRICS_DIR = ROOT / "reports" / "metrics"
LABELS = ["negative", "neutral", "positive"]

SYSTEM_PROMPT = """You label app store reviews by sentiment. Reviews can be in Ukrainian, Russian, English or other languages.
Labels:
- negative: complaint, bug, something does not work, anger, sarcasm, a question that points to a problem
- neutral: a real question, a suggestion or feature request without complaint, a plain fact, praise and complaint in equal measure, praise together with a wish
- positive: praise, satisfaction, short approval like "ok" or "норм"
Judge only the text.
Return only a JSON list with one object per review: {"id": <id>, "sentiment": "negative" | "neutral" | "positive"}"""


def parse_labels(answer):
    start, end = answer.find("["), answer.rfind("]")
    if start == -1 or end == -1:
        return {}
    items = json.loads(answer[start:end + 1])
    return {int(item["id"]): item["sentiment"] for item in items if item.get("sentiment") in LABELS}


def label_batch(texts, provider, retries=3):
    reviews = [{"id": i, "text": text} for i, text in enumerate(texts)]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(reviews, ensure_ascii=False)},
    ]
    for attempt in range(retries):
        try:
            labels = parse_labels(chat(messages, provider=provider))
            return [labels.get(i) for i in range(len(texts))]
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(20 * (attempt + 1))
    return [None] * len(texts)


def label_reviews(df, out_path, provider, batch_size=20, pause=4.0):
    done = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(columns=["reviewId", "llm_sentiment"])
    todo = df[~df["reviewId"].isin(done["reviewId"])]
    print(f"{len(done)} already labeled, {len(todo)} to go")

    labeled = 0
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        labels = label_batch(batch["text"].tolist(), provider)
        rows = pd.DataFrame({"reviewId": batch["reviewId"], "llm_sentiment": labels}).dropna()
        rows.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        labeled += len(rows)
        print(f"{min(start + batch_size, len(todo))}/{len(todo)} processed, {labeled} labeled")
        time.sleep(pause)
    return pd.read_csv(out_path)


def compare_with_gold(gold, llm_labels, reviews, provider):
    df = gold.merge(llm_labels, on="reviewId").merge(reviews[["reviewId", "score"]], on="reviewId")
    df["stars"] = df["score"].map(SENTIMENT)
    print(f"\nLLM vs manual labels (n={len(df)})")
    print(classification_report(df["sentiment"], df["llm_sentiment"], labels=LABELS, digits=3, zero_division=0))

    metrics = {"provider": provider, "model": get_model(provider), "n": len(df)}
    for name in ["llm_sentiment", "stars"]:
        metrics[name] = {
            "accuracy": round(accuracy_score(df["sentiment"], df[name]), 3),
            "kappa": round(cohen_kappa_score(df["sentiment"], df[name]), 3),
            "macro_f1": round(
                classification_report(df["sentiment"], df[name], labels=LABELS, zero_division=0, output_dict=True)["macro avg"]["f1-score"], 3
            ),
        }
        print(f"{name:<15} accuracy {metrics[name]['accuracy']}  kappa {metrics[name]['kappa']}  macro-F1 {metrics[name]['macro_f1']}")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"llm_labels_{provider}.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["gold", "train"], default="gold")
    parser.add_argument("--n", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pause", type=float, default=4.0)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    args = parser.parse_args()
    provider = get_provider(args.provider)

    reviews = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    gold = load_gold().dropna(subset=["sentiment"])
    if args.source == "gold":
        df = gold[["reviewId", "text"]]
    else:
        train = reviews[(reviews["split"] == "train") & ~reviews["reviewId"].isin(gold["reviewId"])]
        df = train.sample(min(args.n, len(train)), random_state=42)[["reviewId", "text"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"labels_{args.source}_{provider}.csv"
    llm_labels = label_reviews(df, out_path, provider, args.batch_size, args.pause)
    if args.source == "gold":
        compare_with_gold(gold, llm_labels, reviews, provider)


if __name__ == "__main__":
    main()