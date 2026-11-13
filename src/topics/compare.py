"""Compare topic modeling approaches on complaints (negative reviews):
BERTopic-style pipeline (multilingual embeddings + UMAP + HDBSCAN + c-TF-IDF),
k-means on the same reduced embeddings, and LDA. Metrics: NPMI coherence,
topic diversity, silhouette (cosine, on the original embeddings).

Usage:
    python -m src.topics.compare
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.experiments import MODEL_PATH as SENTIMENT_PATH
from src.sentiment.threshold import apply_threshold
from src.topics.modeling import (
    EMBEDDING_MODEL,
    cluster_hdbscan,
    cluster_kmeans,
    ctfidf_words,
    diversity,
    embed,
    lda_topics,
    npmi_coherence,
    reduce,
    silhouette,
    word_coherence,
)

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "metrics" / "topics.json"
TOPICS_DIR = ROOT / "reports" / "topics"
MIN_CHARS = 30


def load_complaints():
    df = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    saved = joblib.load(SENTIMENT_PATH)
    model = saved["model"]
    df["pred"] = apply_threshold(model.predict_proba(df["text"]), list(model.classes_), saved["threshold"])
    complaints = df[(df["pred"] == "negative") & (df["text"].str.len() >= MIN_CHARS)]
    return complaints.reset_index(drop=True)


def topics_table(df, labels, words, embeddings, n_examples=3):
    rows = []
    for label, top in words.items():
        idx = np.where(labels == label)[0]
        members = df.iloc[idx]
        centroid = embeddings[idx].mean(axis=0)
        closest = idx[np.argsort(embeddings[idx] @ centroid)[::-1][:n_examples]]
        rows.append({
            "topic": label,
            "size": len(idx),
            "share": round(len(idx) / len(df), 3),
            "words": ", ".join(w.replace("_", " ") for w in top),
            "apps": ", ".join(members["app"].value_counts().head(3).index),
            "examples": " | ".join(df.loc[closest, "text"].str.slice(0, 150)),
        })
    return pd.DataFrame(rows).sort_values("size", ascending=False)


def evaluate(name, texts, embeddings, labels, words, seconds):
    result = {
        "n_topics": len(words),
        "noise_share": round(float((labels == -1).mean()), 3),
        "npmi": round(npmi_coherence(texts, words), 3),
        "word_coherence": round(word_coherence(words), 3),
        "diversity": round(diversity(words), 3),
        "silhouette": round(silhouette(embeddings, labels), 3),
        "time_s": round(seconds, 1),
    }
    print(f"{name:<9} topics {result['n_topics']:<4} noise {result['noise_share']:<6} NPMI {result['npmi']:<7} "
          f"word coherence {result['word_coherence']:<6} diversity {result['diversity']:<6} "
          f"silhouette {result['silhouette']:<7} time {result['time_s']} s")
    return result


def main():
    complaints = load_complaints()
    texts = complaints["text"].tolist()
    print(f"Complaints: {len(texts)} negative reviews with at least {MIN_CHARS} characters")

    start = time.perf_counter()
    embeddings = embed(texts)
    embed_time = time.perf_counter() - start
    start = time.perf_counter()
    reduced = reduce(embeddings)
    reduce_time = time.perf_counter() - start
    print(f"Embeddings {embed_time:.0f} s, UMAP {reduce_time:.0f} s\n")

    results, tables = {}, {}
    start = time.perf_counter()
    labels = cluster_hdbscan(reduced)
    words = ctfidf_words(texts, labels)
    results["bertopic"] = evaluate("BERTopic", texts, embeddings, labels, words, time.perf_counter() - start + reduce_time)
    tables["bertopic"] = topics_table(complaints, labels, words, embeddings)
    k = len(words)

    start = time.perf_counter()
    labels = cluster_kmeans(reduced, k)
    words = ctfidf_words(texts, labels)
    results["kmeans"] = evaluate("k-means", texts, embeddings, labels, words, time.perf_counter() - start + reduce_time)
    tables["kmeans"] = topics_table(complaints, labels, words, embeddings)

    start = time.perf_counter()
    labels, words = lda_topics(texts, k)
    results["lda"] = evaluate("LDA", texts, embeddings, labels, words, time.perf_counter() - start)
    tables["lda"] = topics_table(complaints, labels, words, embeddings)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = {"n_complaints": len(texts), "min_chars": MIN_CHARS, "embedding_model": EMBEDDING_MODEL,
            "embedding_time_s": round(embed_time, 1), "k": k}
    METRICS_PATH.write_text(json.dumps({**meta, "methods": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(TOPICS_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved metrics to {METRICS_PATH}, topic tables to {TOPICS_DIR}")


if __name__ == "__main__":
    main()