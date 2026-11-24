"""Language mix of complaint topics (H2): does a topic join complaints written in
different languages, or does every language get its own topics. The topics are
rebuilt with the same settings as src.topics.compare, so they are the same topics
as in reports/topics.

Usage:
    python -m src.evaluation.topic_languages
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.topics.compare import load_complaints
from src.topics.modeling import cluster_hdbscan, cluster_kmeans, ctfidf_words, embed, lda_topics, reduce

ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "reports" / "metrics" / "topic_languages.json"
TOPICS_DIR = ROOT / "reports" / "topics"
LANGS = ["uk", "ru", "en", "pl", "es"]
CYRILLIC, LATIN = ["uk", "ru"], ["en", "pl", "es"]
MIN_SHARE = 0.10
ONE_LANGUAGE = 0.90
TOP = 15


def topic_table(langs, labels, words):
    df = pd.DataFrame({"lang": langs, "topic": labels})
    df = df[df["topic"] != -1]
    shares = pd.crosstab(df["topic"], df["lang"], normalize="index").reindex(columns=LANGS, fill_value=0.0)
    table = shares.round(3)
    table.insert(0, "size", df["topic"].value_counts())
    table["languages"] = (shares >= MIN_SHARE).sum(axis=1)
    table["cross_script"] = (shares[CYRILLIC].sum(axis=1) >= MIN_SHARE) & (shares[LATIN].sum(axis=1) >= MIN_SHARE)
    table["main_share"] = shares.max(axis=1).round(3)
    table["words"] = [", ".join(w.replace("_", " ") for w in words.get(t, [])) for t in table.index]
    return table.sort_values("size", ascending=False)


def summary(table, langs):
    sizes = table["size"]
    known = pd.Series(langs)[pd.Series(langs).isin(LANGS)]
    mixed = table["languages"] >= 2
    return {
        "topics": len(table),
        "multilingual_topics": round(float(mixed.mean()), 3),
        "complaints_in_multilingual": round(float(sizes[mixed].sum() / sizes.sum()), 3),
        "cross_script_topics": round(float(table["cross_script"].mean()), 3),
        "one_language_topics": round(float((table["main_share"] >= ONE_LANGUAGE).mean()), 3),
        "main_share_weighted": round(float(np.average(table["main_share"], weights=sizes)), 3),
        "main_share_if_random": round(float(known.value_counts(normalize=True).max()), 3),
        "top15_multilingual": round(float(mixed.head(TOP).mean()), 3),
        "top15_cross_script": round(float(table["cross_script"].head(TOP).mean()), 3),
    }


def same_as_saved(name, table):
    path = TOPICS_DIR / f"{name}.csv"
    if not path.exists():
        return None
    saved = pd.read_csv(path)["size"].sort_values(ascending=False)
    return saved[saved > 0].tolist() == table["size"].tolist()


def main():
    complaints = load_complaints()
    texts, langs = complaints["text"].tolist(), complaints["lang"].tolist()
    print(f"Complaints: {len(texts)}")
    print("Languages:", complaints["lang"].value_counts(normalize=True).round(3).to_dict())

    embeddings = embed(texts)
    reduced = reduce(embeddings)
    labels = cluster_hdbscan(reduced)
    words = ctfidf_words(texts, labels)
    k = len(words)
    runs = {"bertopic": (labels, words)}
    labels = cluster_kmeans(reduced, k)
    runs["kmeans"] = (labels, ctfidf_words(texts, labels))
    runs["lda"] = lda_topics(texts, k)

    results = {"complaints": len(texts), "languages": complaints["lang"].value_counts().to_dict(),
               "min_share": MIN_SHARE, "methods": {}}
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (labels, words) in runs.items():
        table = topic_table(langs, labels, words)
        table.to_csv(TOPICS_DIR / f"{name}_languages.csv", encoding="utf-8-sig")
        results["methods"][name] = {**summary(table, langs), "same_topics_as_saved": same_as_saved(name, table)}
        print(f"\n{name}: {results['methods'][name]}")
        print(table.head(5)[["size", *LANGS, "words"]].to_string())

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {METRICS_PATH} and {TOPICS_DIR}/*_languages.csv")


if __name__ == "__main__":
    main()