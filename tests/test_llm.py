import pandas as pd

from src.llm import client, label, synthetic
from src.sentiment.experiments import with_synthetic


def test_parse_labels_handles_extra_text_and_bad_labels():
    answer = 'Sure:\n```json\n[{"id": 0, "sentiment": "negative"}, {"id": 1, "sentiment": "angry"}, {"id": 2, "sentiment": "neutral"}]\n```'
    assert label.parse_labels(answer) == {0: "negative", 2: "neutral"}


def test_parse_labels_without_json():
    assert label.parse_labels("I can't help with that") == {}


def test_label_batch_keeps_order(monkeypatch):
    monkeypatch.setattr(label, "chat", lambda messages, provider=None: '[{"id": 1, "sentiment": "positive"}, {"id": 0, "sentiment": "negative"}]')
    assert label.label_batch(["погано", "добре", "ок"], "openrouter") == ["negative", "positive", None]


def test_label_batch_gives_up_after_retries(monkeypatch):
    def failing_chat(messages, provider=None):
        raise RuntimeError("429")

    monkeypatch.setattr(label, "chat", failing_chat)
    monkeypatch.setattr(label.time, "sleep", lambda seconds: None)
    assert label.label_batch(["a", "b"], "openrouter") == [None, None]


def test_parse_reviews_keeps_only_text():
    answer = '["  Додайте темну тему ", "", 42, "How do I change my phone number?"]'
    assert synthetic.parse_reviews(answer) == ["Додайте темну тему", "How do I change my phone number?"]


def test_provider_and_model_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:12b")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it")
    assert client.get_provider() == "ollama"
    assert client.get_model() == "gemma3:12b"
    assert client.get_model("openrouter") == "google/gemma-4-26b-a4b-it"


def test_with_synthetic_adds_neutral_only_to_training():
    texts, labels = with_synthetic(
        pd.Series(["погано", "добре"]),
        pd.Series(["negative", "positive"]),
        pd.DataFrame({"text": ["Додайте темну тему"]}),
    )
    assert texts.tolist() == ["погано", "добре", "Додайте темну тему"]
    assert labels.tolist() == ["negative", "positive", "neutral"]