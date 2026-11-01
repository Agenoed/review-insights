import pandas as pd

from src.evaluation import gold as gold_module


def test_load_gold(tmp_path, monkeypatch):
    path = tmp_path / "to_label.xlsx"
    pd.DataFrame({
        "reviewId": ["a", "b", "c", "d"],
        "text": ["NA", "добре", "спам", "ще не розмічено"],
        "sentiment": ["neg", "pos", "skip", ""],
        "lang": ["en", "uk", "unclear", ""],
    }).to_excel(path, index=False)
    monkeypatch.setattr(gold_module, "GOLD_PATH", path)

    gold = gold_module.load_gold()

    assert gold["reviewId"].tolist() == ["a", "b", "c"]
    assert gold.loc[gold["reviewId"] == "a", "text"].item() == "NA"
    assert gold["sentiment"].tolist()[:2] == ["negative", "positive"]
    assert pd.isna(gold.loc[gold["reviewId"] == "c", "sentiment"].item())