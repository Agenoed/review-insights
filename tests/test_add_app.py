import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.pipeline import add_app as A

LINK = "https://play.google.com/store/apps/details?id=com.ubercab&hl=uk"


def reviews(ids):
    start = datetime(2026, 11, 20)
    return pd.DataFrame([{
        "reviewId": f"r{i}", "content": "😡😡" if i == 0 else f"водій не приїхав, гроші списали {i}",
        "score": 1, "thumbsUpCount": 0, "reviewCreatedVersion": "1.0", "at": start - timedelta(hours=i),
        "replyContent": None, "repliedAt": None, "app": "uber", "category": "maps",
        "country": "ua", "request_lang": "uk",
    } for i in ids])


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "apps.json").write_text(
        json.dumps({"apps": [{"name": "glovo", "id": "com.glovo", "category": "food"}]}), encoding="utf-8")
    added = tmp_path / "added"
    batches = [reviews(range(0, 6)), reviews(range(3, 9))]
    calls = {"describe": 0}

    def fake_describe(name, provider=None, df=None):
        calls["describe"] += 1
        return {"app": name, "problems": [{"id": 1}, {"id": 2}], "complaints": 5}

    monkeypatch.setattr(A, "ROOT", tmp_path)
    monkeypatch.setattr(A, "PROCESSED_PATH", tmp_path / "no_reviews.csv")
    monkeypatch.setattr(A, "ADDED_DIR", added)
    monkeypatch.setattr(A, "RAW_ADDED_DIR", added / "raw")
    monkeypatch.setattr(A, "REGISTRY_PATH", added / "apps.json")
    monkeypatch.setattr(A, "OUT_DIR", tmp_path / "apps")
    monkeypatch.setattr(A, "play_app", lambda app_id, lang, country: {"title": "Uber - Request a ride", "genreId": "MAPS"})
    monkeypatch.setattr(A, "collect", lambda config, per_lang, pause: batches.pop(0))
    monkeypatch.setattr(A, "detect_lang", lambda text: "uk" if any(ch.isalpha() for ch in text) else "no_letters")
    monkeypatch.setattr(A, "predict", lambda texts: (np.array(["negative"] * len(texts), dtype=object),
                                                    np.full(len(texts), 0.9), "fake"))
    monkeypatch.setattr(A, "load_reviews", lambda: pd.DataFrame())
    monkeypatch.setattr(A, "describe_app", fake_describe)
    monkeypatch.setattr(A, "build_report", lambda name, provider=None: ({"app": name}, "# Звіт"))
    return {"dir": tmp_path, "calls": calls}


def test_parse_link():
    assert A.parse_link(LINK) == "com.ubercab"
    assert A.parse_link(" ee.mtakso.client ") == "ee.mtakso.client"
    for bad in ["https://apps.apple.com/us/app/venmo/id351727428", "uber", ""]:
        with pytest.raises(ValueError):
            A.parse_link(bad)


def test_names():
    assert A.default_name("ПУМБ : Онлайн Банк", "com.fuib.android.spot.online") == "pumb"
    assert A.default_name("Дія", "ua.gov.diia.app") == "diia"
    assert A.default_name("★", "com.x.bestapp") == "bestapp"
    assert A.default_name("Google Chrome: Fast & Secure", "com.android.chrome") == "google_chrome"
    assert A.default_name("TikTok - Videos, Music & LIVE", "com.zhiliaoapp.musically") == "tiktok"
    assert A.default_name("Subway Surfers", "com.kiloo.subwaysurf") == "subway_surfers"
    assert A.clean_name(" Uber App ") == "uber_app"
    assert A.clean_name("Укрзалізниця") == "ukrzaliznytsia"


def test_add_app_saves_everything(sandbox):
    lines = []
    result = A.add_app(LINK, log=lines.append, pause=0)
    out = sandbox["dir"]
    assert result["name"] == "uber"
    assert result["reviews"] == 5
    assert result["problems"] == 2 and result["llm_error"] is None
    assert (out / "added" / "uber.csv").exists()
    assert len(list((out / "added" / "raw").glob("uber_*.csv"))) == 1
    for end in ("_described.json", "_report.json", "_report.md"):
        assert (out / "apps" / f"uber{end}").exists()
    described = json.loads((out / "apps" / "uber_described.json").read_text(encoding="utf-8"))
    assert described["title"] == "Uber - Request a ride"
    registry = json.loads((out / "added" / "apps.json").read_text(encoding="utf-8"))
    assert registry["uber"]["id"] == "com.ubercab"
    assert lines[0].startswith("Uber - Request a ride (com.ubercab)")


def test_update_adds_only_new_reviews(sandbox):
    A.add_app(LINK, use_llm=False, pause=0)
    result = A.add_app(LINK, use_llm=False, pause=0)
    df = pd.read_csv(sandbox["dir"] / "added" / "uber.csv")
    assert result["reviews"] == 8
    assert df["reviewId"].is_unique
    assert sandbox["calls"]["describe"] == 0


def test_llm_error_keeps_reviews(sandbox, monkeypatch):
    def fail(name, provider=None, df=None):
        raise ValueError("too few complaints for uber: 3")

    monkeypatch.setattr(A, "describe_app", fail)
    result = A.add_app(LINK, pause=0)
    assert result["llm_error"] == "замало скарг для пошуку проблем"
    assert result["reviews"] == 5
    assert not (sandbox["dir"] / "apps" / "uber_described.json").exists()


@pytest.mark.parametrize("link, kwargs, message", [
    ("com.glovo", {}, "основному наборі"),
    (LINK, {"name": "glovo"}, "уже зайнята"),
    (LINK, {"langs": ["de"]}, "Мови"),
    (LINK, {"country": "ukr"}, "Країна"),
    (LINK, {"name": "!"}, "Назва"),
])
def test_check_rejects(sandbox, link, kwargs, message):
    with pytest.raises(ValueError, match=message):
        A.add_app(link, pause=0, **kwargs)


def test_same_app_under_other_name(sandbox):
    A.add_app(LINK, use_llm=False, pause=0)
    with pytest.raises(ValueError, match="під назвою uber"):
        A.add_app(LINK, name="taxi", use_llm=False, pause=0)


def test_remove_app(sandbox):
    A.add_app(LINK, pause=0)
    other = sandbox["dir"] / "added" / "raw" / "uber_app_20261124_0101.csv"
    other.write_text("x", encoding="utf-8")
    info = A.remove_app("uber")
    out = sandbox["dir"]
    assert info["title"] == "Uber - Request a ride"
    assert not (out / "added" / "uber.csv").exists()
    assert not list((out / "apps").glob("uber_*"))
    assert other.exists()
    assert A.read_registry() == {}
    with pytest.raises(ValueError):
        A.remove_app("glovo")