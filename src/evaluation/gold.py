from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "labeling" / "to_label.xlsx"
LABELS = {"neg": "negative", "neu": "neutral", "pos": "positive"}


def load_gold():
    gold = pd.read_excel(GOLD_PATH, keep_default_na=False)
    gold = gold[(gold["sentiment"] != "") & (gold["lang"] != "")].copy()
    gold["sentiment"] = gold["sentiment"].str.strip().map(LABELS)
    gold["lang"] = gold["lang"].str.strip()
    return gold