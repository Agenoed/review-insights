from types import SimpleNamespace

from src.data import collect_google_play as collect_module


def fake_reviews(total):
    def _reviews(app_id, lang, country, sort, count, continuation_token=None):
        start = continuation_token.position if continuation_token else 0
        end = min(start + count, total)
        batch = [{"reviewId": f"{app_id}-{i}", "content": f"text {i}", "score": 5} for i in range(start, end)]
        return batch, SimpleNamespace(position=end, token=None if end >= total else "next")
    return _reviews


def test_fetch_respects_limit(monkeypatch):
    monkeypatch.setattr(collect_module, "reviews", fake_reviews(total=1000))
    assert len(collect_module.fetch_reviews("app", "uk", "ua", limit=450, pause=0)) == 450


def test_fetch_stops_when_reviews_end(monkeypatch):
    monkeypatch.setattr(collect_module, "reviews", fake_reviews(total=120))
    assert len(collect_module.fetch_reviews("app", "uk", "ua", limit=500, pause=0)) == 120


def test_collect_drops_duplicates_empty_and_user_names(monkeypatch):
    def fake_fetch(app_id, lang, country, limit, pause=1.0):
        return [
            {"reviewId": "same", "content": "добре", "score": 5, "userName": "Іван"},
            {"reviewId": "empty", "content": "   ", "score": 1, "userName": "Петро"},
        ]

    monkeypatch.setattr(collect_module, "fetch_reviews", fake_fetch)
    config = {"country": "ua", "languages": ["uk", "ru"], "apps": [{"name": "a", "id": "a", "category": "bank"}]}
    df = collect_module.collect(config, per_lang=10, pause=0)

    assert df["reviewId"].tolist() == ["same"]
    assert "userName" not in df.columns