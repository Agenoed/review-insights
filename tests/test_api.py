import json

import pytest
from fastapi.testclient import TestClient

from src.api import main as M
from src.pipeline import add_app as A

DESCRIBED = {
    "app": "glovo", "period_days": 30, "period_end": "2026-11-10", "reviews": 856, "negative_share": 0.49,
    "complaints": 3, "other_share": 0.1, "several_share": 0.3, "trend": None, "llm_model": "test",
    "problems": [{"id": 3, "title": "Підтримка", "description": "Не відповідають.", "complaints": 2, "share": 0.67,
                  "direction": None, "trend": None, "quotes": ["стара"], "related": []}],
    "items": [
        {"id": "a", "at": "2026-11-01", "text": "стара", "current": True, "problems": [3]},
        {"id": "b", "at": "2026-11-09", "text": "нова", "current": True, "problems": [3]},
        {"id": "c", "at": "2026-10-01", "text": "минулий період", "current": False, "problems": [3]},
        {"id": "d", "at": "2026-11-05", "text": "без проблеми", "current": True, "problems": []},
    ],
}
REGISTRY = {"uber": {"id": "com.ubercab", "title": "Uber", "link": "https://play.google.com/store/apps/details?id=com.ubercab",
                     "country": "ua", "languages": ["uk"], "per_lang": 100, "llm_error": None}}


@pytest.fixture
def api(tmp_path, monkeypatch):
    apps, added = tmp_path / "apps", tmp_path / "added"
    apps.mkdir()
    added.mkdir()
    (apps / "glovo_described.json").write_text(json.dumps(DESCRIBED, ensure_ascii=False), encoding="utf-8")
    (apps / "glovo_report.md").write_text("# Звіт\n", encoding="utf-8")
    (apps / "glovo_report.json").write_text("{}", encoding="utf-8")
    (added / "apps.json").write_text(json.dumps(REGISTRY), encoding="utf-8")
    (added / "uber.csv").write_text("reviewId\nr1\n", encoding="utf-8")
    calls = []

    def fake_add(**request):
        log = request.pop("log")
        calls.append(request)
        if request["link"].endswith("broken"):
            raise ValueError("Не вдалося знайти застосунок")
        log("крок 1")
        return {"name": request["name"] or "new", "reviews": 10}

    monkeypatch.setattr(M, "OUT_DIR", apps)
    monkeypatch.setattr(M, "JOBS", {})
    monkeypatch.setattr(M, "add_app", fake_add)
    monkeypatch.setattr(M, "detect_lang", lambda text: "uk")
    monkeypatch.setattr(M, "predict", lambda texts: (["negative"] * len(texts), [0.91] * len(texts), "fake"))
    monkeypatch.setattr(A, "OUT_DIR", apps)
    monkeypatch.setattr(A, "ADDED_DIR", added)
    monkeypatch.setattr(A, "RAW_ADDED_DIR", added / "raw")
    monkeypatch.setattr(A, "REGISTRY_PATH", added / "apps.json")
    return TestClient(M.app), calls


def test_health(api):
    client, _ = api
    assert client.get("/health").json() == {"status": "ok"}


def test_sentiment(api):
    client, _ = api
    response = client.post("/sentiment", json={"texts": ["погано", "добре"]})
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake"
    assert body["results"][0] == {"text": "погано", "sentiment": "negative", "confidence": 0.91, "lang": "uk"}
    assert client.post("/sentiment", json={"texts": []}).status_code == 422
    bad = client.post("/sentiment", content="{\"texts\": [\"a\" \"b\"]}", headers={"Content-Type": "application/json"})
    assert bad.status_code == 422


def test_apps_list(api):
    client, _ = api
    apps = {a["name"]: a for a in client.get("/apps").json()}
    assert set(apps) == {"glovo", "uber"}
    assert apps["glovo"]["problems"] == 1 and apps["glovo"]["report"] and not apps["glovo"]["added"]
    assert apps["uber"]["added"] and apps["uber"]["title"] == "Uber" and apps["uber"]["problems"] is None


def test_problems(api):
    client, _ = api
    body = client.get("/apps/glovo/problems").json()
    assert body["title"] == "glovo"
    assert body["problems"][0]["title"] == "Підтримка"
    assert "items" not in body
    assert client.get("/apps/nothing/problems").status_code == 404
    assert client.get("/apps/Bad-Name/problems").status_code == 400


def test_one_problem_with_complaints(api):
    client, _ = api
    body = client.get("/apps/glovo/problems/3").json()
    assert [c["text"] for c in body["complaints_list"]] == ["нова", "стара"]
    assert len(client.get("/apps/glovo/problems/3", params={"limit": 1}).json()["complaints_list"]) == 1
    assert client.get("/apps/glovo/problems/99").status_code == 404


def test_report(api):
    client, _ = api
    assert client.get("/apps/glovo/report").json()["markdown"] == "# Звіт\n"
    assert client.get("/apps/uber/report").status_code == 404


def test_add_runs_job(api):
    client, calls = api
    response = client.post("/apps", json={"link": "com.ubercab", "country": "PL"})
    assert response.status_code == 202
    job = client.get(response.json()["status_url"]).json()
    assert job["status"] == "done"
    assert job["log"] == ["крок 1"]
    assert job["result"]["reviews"] == 10
    assert calls[0]["country"] == "pl" and calls[0]["langs"] == ["pl", "en"]


def test_add_rejects_bad_input(api):
    client, calls = api
    assert client.post("/apps", json={"link": "https://apps.apple.com/app/id1"}).status_code == 400
    assert client.post("/apps", json={"link": "com.ubercab", "langs": ["de"]}).status_code == 400
    assert client.post("/apps", json={"link": "com.ubercab", "per_lang": 50}).status_code == 422
    assert client.get("/jobs/nope").status_code == 404
    assert calls == []


def test_failed_job(api):
    client, _ = api
    response = client.post("/apps", json={"link": "com.broken"})
    job = client.get(response.json()["status_url"]).json()
    assert job["status"] == "error"
    assert "Не вдалося" in job["error"]


def test_update_uses_registry(api):
    client, calls = api
    assert client.post("/apps/nothing/update").status_code == 404
    assert client.post("/apps/uber/update").status_code == 202
    assert calls[-1] == {"link": REGISTRY["uber"]["link"], "name": "uber", "country": "ua", "langs": ["uk"],
                         "per_lang": 100, "use_llm": True}


def test_delete(api):
    client, _ = api
    assert client.delete("/apps/glovo").status_code == 404
    M.JOBS["busy"] = {"id": "busy", "status": "running"}
    assert client.delete("/apps/uber").status_code == 409
    M.JOBS["busy"]["status"] = "done"
    response = client.delete("/apps/uber")
    assert response.status_code == 200
    assert response.json() == {"removed": "uber", "title": "Uber"}
    assert "uber" not in {a["name"] for a in client.get("/apps").json()}