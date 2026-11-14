"""Topics of complaints for one app: groups negative reviews of the chosen period
together with the previous period of the same length, attaches outliers to the
nearest topic and shows how each topic changed. Trends are computed only on
review groups (country + language) that were collected for both periods.

Usage:
    python -m src.topics.app_topics --app glovo
    python -m src.topics.app_topics --app monobank --days 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline.score import PREDICTIONS_PATH
from src.preprocessing.prepare import PROCESSED_PATH
from src.topics.modeling import (
    attach_outliers,
    cluster_hdbscan,
    ctfidf_words,
    embed,
    reduce,
    representative,
    shorten,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "apps"
MIN_CHARS = 30
MIN_COMPLAINTS = 150
MIN_TOPICS = 4
GROUP = ["country", "request_lang"]


def load_reviews():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["at"], keep_default_na=False)
    predictions = pd.read_csv(PREDICTIONS_PATH, keep_default_na=False)
    return df.merge(predictions, on="reviewId")


def is_complaint(df):
    return (df["pred"] == "negative") & (df["text"].str.len() >= MIN_CHARS)


def pick_days(app_df, end):
    complaints = app_df[is_complaint(app_df)]
    for days in [30, 60, 90, 180, 365]:
        if (complaints["at"] > end - pd.Timedelta(days=days)).sum() >= MIN_COMPLAINTS:
            return days
    return 365


def collected_since(app_df, since):
    if set(GROUP) <= set(app_df.columns):
        first = app_df.groupby(GROUP)["at"].transform("min")
    else:
        first = pd.Series(app_df["at"].min(), index=app_df.index)
    return (first <= since).to_numpy()


def share(part, total):
    return round(part / total, 3) if total else None


def find_app_topics(df, app, days=None, n_examples=25):
    app_df = df[df["app"] == app].copy()
    if app_df.empty:
        raise ValueError(f"no reviews for {app}")
    end = app_df["at"].max()
    days = days or pick_days(app_df, end)
    start, prev_start = end - pd.Timedelta(days=days), end - pd.Timedelta(days=2 * days)
    app_df["in_trend"] = collected_since(app_df, prev_start)

    current_all = app_df[app_df["at"] > start]
    trend_all = app_df[app_df["in_trend"] & (app_df["at"] > prev_start)]
    trend_current, trend_previous = trend_all[trend_all["at"] > start], trend_all[trend_all["at"] <= start]

    complaints = app_df[is_complaint(app_df) & (app_df["at"] > prev_start)].reset_index(drop=True)
    if len(complaints) < 30:
        raise ValueError(f"too few complaints for {app}: {len(complaints)}")
    is_current = (complaints["at"] > start).to_numpy()
    in_trend = complaints["in_trend"].to_numpy()

    embeddings = embed(complaints["text"], show_progress=False)
    reduced = reduce(embeddings)
    min_size = max(10, len(complaints) // 40)
    labels = cluster_hdbscan(reduced, min_cluster_size=min_size, min_samples=5)
    if len(set(labels) - {-1}) < MIN_TOPICS:
        labels = cluster_hdbscan(reduced, min_cluster_size=min_size, min_samples=5, method="leaf")
    noise_before = float(((labels == -1) & is_current).sum() / max(is_current.sum(), 1))
    labels = attach_outliers(embeddings, labels)
    words = ctfidf_words(complaints["text"], labels, min_df=3)

    n_current = int(is_current.sum())
    n_trend_current = int((is_current & in_trend).sum())
    n_trend_previous = int((~is_current & in_trend).sum())
    has_trend = n_trend_previous > 0 and n_trend_current > 0

    topics = []
    for label, top in words.items():
        idx = np.where(labels == label)[0]
        current_idx = idx[is_current[idx]]
        pool = current_idx if len(current_idx) >= 5 else idx
        topic = {
            "topic": int(label),
            "complaints": len(current_idx),
            "share": share(len(current_idx), n_current),
            "words": [w.replace("_", " ") for w in top],
            "examples": [shorten(text) for text in complaints.loc[representative(embeddings, pool, n_examples), "text"]],
            "trend": None,
        }
        if has_trend:
            now = int((is_current & in_trend)[idx].sum())
            before = int((~is_current & in_trend)[idx].sum())
            topic["trend"] = {"share": share(now, n_trend_current), "previous_share": share(before, n_trend_previous)}
        topics.append(topic)
    topics.sort(key=lambda t: t["complaints"], reverse=True)

    trend = None
    if has_trend:
        trend = {
            "groups": sorted(trend_all[GROUP].drop_duplicates().astype(str).agg("/".join, axis=1)) if set(GROUP) <= set(trend_all.columns) else None,
            "negative_share": share(int((trend_current["pred"] == "negative").sum()), len(trend_current)),
            "previous_negative_share": share(int((trend_previous["pred"] == "negative").sum()), len(trend_previous)),
            "complaints": n_trend_current,
            "previous_complaints": n_trend_previous,
        }

    return {
        "app": app,
        "period_days": days,
        "period_end": end.date().isoformat(),
        "reviews": len(current_all),
        "negative_share": share(int((current_all["pred"] == "negative").sum()), len(current_all)),
        "complaints": n_current,
        "noise_before_attach": round(noise_before, 3),
        "other_share": share(int(((labels == -1) & is_current).sum()), n_current),
        "trend": trend,
        "topics": topics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    result = find_app_topics(load_reviews(), args.app, args.days)
    print(f"{result['app']}: last {result['period_days']} days up to {result['period_end']}, "
          f"{result['reviews']} reviews, negative {result['negative_share']:.0%}")
    print(f"Complaints: {result['complaints']}, {len(result['topics'])} topics, "
          f"noise {result['noise_before_attach']:.0%} before attaching, outside topics {result['other_share']:.0%}")
    trend = result["trend"]
    if trend:
        print(f"Trend on {', '.join(trend['groups'] or [])}: negative {trend['previous_negative_share']:.0%} -> "
              f"{trend['negative_share']:.0%}, complaints {trend['previous_complaints']} -> {trend['complaints']}")
    else:
        print("Trend: not available, the previous period was not fully collected")
    print()
    for t in result["topics"][:10]:
        change = f"{t['trend']['previous_share']:.0%} -> {t['trend']['share']:.0%}" if t["trend"] else "no trend"
        print(f"{t['complaints']:>4} ({t['share']:.0%}), {change:<12} {', '.join(t['words'][:6])}")
    print("\nExample:", result["topics"][0]["examples"][0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.app}_topics.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()