from src.llm.report import complaints_word, links, markets_text, percent, period_text, trend_text
from src.llm.topics import count, good_text, interleave, to_indices, trend_direction
from src.topics.app_topics import markets
import pandas as pd


def test_to_indices_keeps_valid_numbers_once():
    assert to_indices([1, "2", 2, 0, 7, "x", None], 5) == [1, 2]
    assert to_indices("1, 2", 5) == []


def test_interleave_takes_items_by_turns():
    assert interleave([[1, 2, 3], [4]], 10) == [1, 4, 2, 3]
    assert interleave([[1, 2, 3], [4]], 2) == [1, 4]
    assert interleave([], 5) == []


def test_good_text_needs_ukrainian_without_foreign_letters():
    assert good_text({"title": "Не приходить код", "description": "Користувачі не отримують код у SMS."})
    assert not good_text({"title": "Не приходит код", "description": "Пользователи не получают код ы."})
    assert not good_text({"title": "Помилка", "description": "Помилка оплати"})
    assert not good_text({"title": "Помилка ไทย", "description": "Користувачі пишуть про помилку."})
    assert not good_text({"title": "Помилка"})


def test_trend_direction():
    totals = {"complaints": 100, "previous_complaints": 100}
    assert trend_direction(None, totals) is None
    assert trend_direction({"complaints": 4, "previous_complaints": 3}, totals) == "stable"
    assert trend_direction({"complaints": 60, "previous_complaints": 20}, totals) == "growing"
    assert trend_direction({"complaints": 20, "previous_complaints": 60}, totals) == "declining"
    assert trend_direction({"complaints": 22, "previous_complaints": 20}, totals) == "stable"


def test_count_shares_and_related():
    problems = [{"id": 1, "groups": [0]}, {"id": 2, "groups": [1]}]
    items = [{"text": f"скарга {k}", "current": True, "in_trend": True, "problems": [1, 2]} for k in range(4)]
    items += [{"text": "інша скарга", "current": True, "in_trend": True, "problems": []},
              {"text": "стара скарга", "current": False, "in_trend": True, "problems": [1]}]
    clusters = [{"topic": 0, "examples": ["скарга 0"]}, {"topic": 1, "examples": ["скарга 1"]}]
    count(problems, items, clusters, {"complaints": 5, "previous_complaints": 1})
    first = problems[0]
    assert first["complaints"] == 4
    assert first["share"] == 0.8
    assert first["trend"]["previous_complaints"] == 1
    assert first["related"] == [{"id": 2, "complaints": 4}]
    assert first["quotes"][0] == "скарга 0"


def test_complaints_word():
    assert [complaints_word(n) for n in (1, 2, 5, 11, 21, 22, 112)] == \
        ["скарга", "скарги", "скарг", "скарг", "скарга", "скарги", "скарг"]
    assert percent(0.373) == "37%"


def test_period_text_mentions_short_data():
    result = {"period_days": 365, "period_end": "2026-11-23", "first_day": "2026-10-16"}
    assert period_text(result) == "останні 365 днів, до 2026-11-23 (відгуки є лише з 2026-10-16)"
    result["first_day"] = "2025-11-25"
    assert period_text(result) == "останні 365 днів, до 2026-11-23"
    assert period_text({"period_days": 30, "period_end": "2026-11-10"}) == "останні 30 днів, до 2026-11-10"


def test_markets():
    df = pd.DataFrame({"country": ["ua"] * 6 + ["pl", ""], "request_lang": ["uk", "uk", "uk", "ru", "ru", "en", "pl", ""]})
    found = markets(df)
    assert found == {"ua": {"uk": 3, "ru": 2, "en": 1}, "pl": {"pl": 1}}
    assert markets_text(found) == "Україна (uk, ru, en); Польща (pl)"
    assert markets_text({"de": {"en": 1}}) == "DE (en)"


def test_trend_text():
    trend = {"complaints": 16, "previous_complaints": 11, "share": 0.16, "previous_share": 0.11}
    assert trend_text({"trend": trend, "direction": "growing"}) == "Росте: з 11% до 16% скарг."
    assert "незначуща" in trend_text({"trend": trend, "direction": "stable"})
    assert trend_text({"trend": None, "direction": None}) == ""


def test_links_only_strong_pairs_once():
    problems = [
        {"id": 1, "title": "Підтримка", "complaints": 40, "related": [{"id": 2, "complaints": 6}, {"id": 3, "complaints": 4}]},
        {"id": 2, "title": "Списання", "complaints": 10, "related": [{"id": 1, "complaints": 6}]},
        {"id": 3, "title": "Кур'єри", "complaints": 8, "related": [{"id": 1, "complaints": 4}]},
    ]
    assert links(problems) == ["60% скарг про «Списання» (6 з 10) також стосуються проблеми «Підтримка»."]