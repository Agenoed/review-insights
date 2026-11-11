"""Generate synthetic neutral reviews with an LLM, for training only.
Every generated review is checked by the same LLM labeler as real reviews,
only reviews labeled neutral are used later.

Usage:
    python -m src.llm.synthetic --per-combo 6
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

from src.llm.client import chat, get_model, get_provider
from src.llm.label import label_batch

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "synthetic" / "neutral.csv"

CATEGORIES = {
    "bank": "a mobile banking app",
    "delivery": "a parcel delivery service app",
    "marketplace": "an online marketplace app",
    "food_delivery": "a food delivery app",
    "taxi": "a taxi app",
    "government": "a government services app",
    "telecom": "a mobile operator app",
    "transport": "a railway tickets app",
    "retail": "a supermarket app",
    "health": "a doctor appointments app",
}
LANGUAGES = {"uk": "Ukrainian", "ru": "Russian", "en": "English", "pl": "Polish", "es": "Spanish"}
TYPES = {
    "question": "a real question about how something works, without complaining",
    "request": "a suggestion or feature request, without complaining",
    "fact": "a plain neutral statement of fact about using the app",
    "mixed": "praise and a small wish in equal measure",
}

PROMPT = """Write {n} different short user reviews of {app} in {language}.
Each review is {kind}. The overall sentiment must be neutral: not a complaint and not pure praise.
Write like real users in an app store: different lengths (from a few words to 2-3 sentences), informal style, sometimes small typos, no brand names.
Return only a JSON list of strings."""


def parse_reviews(answer):
    start, end = answer.find("["), answer.rfind("]")
    if start == -1 or end == -1:
        return []
    items = json.loads(answer[start:end + 1])
    return [text.strip() for text in items if isinstance(text, str) and text.strip()]


def generate(category, lang, kind, n, provider):
    prompt = PROMPT.format(n=n, app=CATEGORIES[category], language=LANGUAGES[lang], kind=TYPES[kind])
    for attempt in range(3):
        try:
            answer = chat([{"role": "user", "content": prompt}], provider=provider, temperature=0.9)
            return parse_reviews(answer)
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(20 * (attempt + 1))
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-combo", type=int, default=6)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    parser.add_argument("--pause", type=float, default=1.0)
    args = parser.parse_args()
    provider = get_provider(args.provider)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame(columns=["category", "lang", "type"])
    done_combos = set(zip(done["category"], done["lang"], done["type"]))
    combos = [c for c in itertools.product(CATEGORIES, LANGUAGES, TYPES) if c not in done_combos]
    print(f"{len(done_combos)} combinations done, {len(combos)} to go")

    for i, (category, lang, kind) in enumerate(combos, 1):
        texts = generate(category, lang, kind, args.per_combo, provider)
        if not texts:
            print(f"{i}/{len(combos)} {category}/{lang}/{kind}: nothing generated")
            continue
        rows = pd.DataFrame({
            "text": texts,
            "llm_check": label_batch(texts, provider),
            "category": category,
            "lang": lang,
            "type": kind,
            "model": get_model(provider),
        })
        rows.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)
        neutral = (rows["llm_check"] == "neutral").sum()
        print(f"{i}/{len(combos)} {category}/{lang}/{kind}: {len(texts)} generated, {neutral} neutral")
        time.sleep(args.pause)

    df = pd.read_csv(OUT_PATH).drop_duplicates("text")
    kept = df[df["llm_check"] == "neutral"]
    print(f"\nTotal {len(df)} unique, {len(kept)} confirmed neutral ({len(kept) / len(df):.0%})")
    print(kept.groupby("lang").size().to_string())


if __name__ == "__main__":
    main()