"""Collect Google Play reviews for the apps listed in a config file.

Usage:
    python -m src.data.collect_google_play --per-lang 500
    python -m src.data.collect_google_play --config configs/apps_extra.json --per-lang 1000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "apps.json"
RAW_DIR = ROOT / "data" / "raw"

FIELDS = [
    "reviewId",
    "content",
    "score",
    "thumbsUpCount",
    "reviewCreatedVersion",
    "at",
    "replyContent",
    "repliedAt",
]


def fetch_reviews(app_id, lang, country, limit, batch_size=200, pause=1.0):
    result = []
    token = None
    while len(result) < limit:
        batch, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=min(batch_size, limit - len(result)),
            continuation_token=token,
        )
        if not batch:
            break
        result.extend(batch)
        if getattr(token, "token", None) is None:
            break
        time.sleep(pause)
    return result[:limit]


def markets(app, config):
    return app.get("markets", [{"country": config["country"], "languages": config["languages"]}])


def collect(config, per_lang, pause):
    rows = []
    for app in config["apps"]:
        for market in markets(app, config):
            country = market["country"]
            for lang in market["languages"]:
                try:
                    batch = fetch_reviews(app["id"], lang, country, per_lang, pause=pause)
                except Exception as e:
                    print(f"{app['name']} ({country}/{lang}): failed - {e}")
                    continue
                for r in batch:
                    row = {f: r.get(f) for f in FIELDS}
                    row["app"] = app["name"]
                    row["category"] = app["category"]
                    row["country"] = country
                    row["request_lang"] = lang
                    rows.append(row)
                print(f"{app['name']} ({country}/{lang}): {len(batch)}")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("reviewId")
    df = df[df["content"].fillna("").str.strip() != ""]
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--per-lang", type=int, default=500)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    df = collect(config, args.per_lang, args.pause)
    if df.empty:
        print("No reviews collected")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"google_play_{args.config.stem}_{datetime.now():%Y%m%d}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} reviews to {out}")


if __name__ == "__main__":
    main()