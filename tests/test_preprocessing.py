import pandas as pd

from src.preprocessing.prepare import add_split, clean_text, make_labeling_sample


def make_df():
    sentiments = ["negative"] * 400 + ["neutral"] * 60 + ["positive"] * 540
    return pd.DataFrame({
        "reviewId": [f"id{i}" for i in range(1000)],
        "text": ["text"] * 1000,
        "sentiment": sentiments,
    })


def test_clean_text_removes_urls_and_extra_spaces():
    assert clean_text("дивіться  https://example.com/a?b=1 тут\n\nвсе") == "дивіться тут все"


def test_clean_text_keeps_case_and_emoji():
    assert clean_text("Супер 👍") == "Супер 👍"


def test_split_proportions():
    counts = add_split(make_df())["split"].value_counts()
    assert counts["train"] == 800
    assert counts["val"] == 100
    assert counts["test"] == 100


def test_split_is_stratified():
    df = add_split(make_df())
    neutral_share = df.groupby("split")["sentiment"].apply(lambda s: (s == "neutral").mean())
    assert neutral_share.between(0.05, 0.07).all()


def test_split_is_reproducible():
    assert add_split(make_df())["split"].tolist() == add_split(make_df())["split"].tolist()


def test_labeling_sample_hides_stars_and_language():
    df = add_split(make_df()).assign(score=1, lang="uk")
    sample = make_labeling_sample(df, n=50)
    assert list(sample.columns) == ["reviewId", "text", "sentiment", "lang"]
    assert (sample["lang"] == "").all()
    assert set(sample["reviewId"]) <= set(df.loc[df["split"] == "test", "reviewId"])