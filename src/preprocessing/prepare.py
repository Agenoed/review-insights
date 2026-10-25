"""Merge raw Google Play dumps, clean text, detect language, add weak labels and split.

Usage:
    python -m src.preprocessing.prepare
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.language import detect_lang

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "reviews.csv"
LABELING_PATH = ROOT / "data" / "labeling" / "to_label.xlsx"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
SPACES_RE = re.compile(r"\s+")
SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def load_raw():
    files = sorted(RAW_DIR.glob("google_play_*.csv"))
    df = pd.concat([pd.read_csv(f, parse_dates=["at"]) for f in files], ignore_index=True)
    df = df.drop_duplicates("reviewId", keep="last").reset_index(drop=True)
    print(f"{len(files)} raw files, {len(df)} unique reviews")
    return df


def clean_text(text):
    text = URL_RE.sub(" ", str(text))
    return SPACES_RE.sub(" ", text).strip()


def add_split(df, seed=42):
    train, rest = train_test_split(
        df, test_size=0.2, stratify=df["sentiment"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["sentiment"], random_state=seed
    )
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def make_labeling_sample(df, n=1000, seed=42):
    test = df[df["split"] == "test"]
    sample = test.sample(min(n, len(test)), random_state=seed)
    return sample[["reviewId", "text"]].assign(sentiment="", lang="")


def main():
    df = load_raw()
    df["text"] = df["content"].map(clean_text)
    df["lang"] = df["text"].map(detect_lang)
    df = df[df["lang"] != "no_letters"].reset_index(drop=True)
    df["sentiment"] = df["score"].map(SENTIMENT)
    df = add_split(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} reviews to {PROCESSED_PATH}")
    print(df["split"].value_counts().to_string())
    print(df["lang"].value_counts(normalize=True).round(3).to_string())

    if LABELING_PATH.exists():
        print(f"{LABELING_PATH.name} already exists, skipping")
        return
    LABELING_PATH.parent.mkdir(parents=True, exist_ok=True)
    make_labeling_sample(df).to_excel(LABELING_PATH, index=False)
    print(f"Labeling sample saved to {LABELING_PATH}")


if __name__ == "__main__":
    main()