"""Checks on the manually labeled sample: do stars match the text, which language detector works best.

Usage:
    python -m src.evaluation.checks
"""

import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.metrics import accuracy_score, classification_report

from src.evaluation.gold import load_gold
from src.preprocessing.language import LINGUA_CODES, detect_lang, detector, lang_by_letters
from src.preprocessing.prepare import PROCESSED_PATH, SENTIMENT

DetectorFactory.seed = 0


def langdetect_only(text):
    try:
        lang = detect(text)
    except LangDetectException:
        return "unknown"
    return lang if lang in {"uk", "ru", "en", "pl"} else "other"


def lingua_only(text):
    result = detector.detect_language_of(text)
    return LINGUA_CODES[result] if result else "unknown"


def letters_langdetect(text):
    return lang_by_letters(text) or langdetect_only(text)


def check_stars(gold, reviews):
    df = gold.dropna(subset=["sentiment"]).merge(reviews[["reviewId", "score"]], on="reviewId")
    df["stars"] = df["score"].map(SENTIMENT)
    print(f"Stars vs manual labels (n={len(df)})")
    print(f"agreement: {accuracy_score(df['sentiment'], df['stars']):.3f}\n")
    print(pd.crosstab(df["sentiment"], df["stars"], rownames=["manual"], colnames=["stars"]))
    print()
    print(classification_report(df["sentiment"], df["stars"], digits=3, zero_division=0))


def check_languages(gold):
    df = gold[gold["lang"].isin(["uk", "ru", "en", "mixed"])]
    no_mixed = df["lang"] != "mixed"
    detectors = {
        "langdetect": langdetect_only,
        "lingua": lingua_only,
        "letters + langdetect": letters_langdetect,
        "letters + lingua": detect_lang,
    }
    print(f"Language detection accuracy (n={len(df)}, without mixed n={no_mixed.sum()})")
    print(f"{'detector':<22}{'all':>8}{'no mixed':>10}")
    for name, func in detectors.items():
        pred = df["text"].map(func)
        acc_all = accuracy_score(df["lang"], pred)
        acc_no_mixed = accuracy_score(df.loc[no_mixed, "lang"], pred[no_mixed])
        print(f"{name:<22}{acc_all:>8.3f}{acc_no_mixed:>10.3f}")


def main():
    gold = load_gold()
    reviews = pd.read_csv(PROCESSED_PATH, keep_default_na=False)
    check_stars(gold, reviews)
    check_languages(gold)


if __name__ == "__main__":
    main()