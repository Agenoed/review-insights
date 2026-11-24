"""REST API over the same pipeline as the dashboard: sentiment of any text, apps with
their complaint problems and reports, adding, updating and removing apps.

Adding an app takes a few minutes, so POST /apps starts a job and returns its id.
The job state is at GET /jobs/{job_id}. Jobs run one at a time and live in memory,
so they are lost when the server restarts (the added apps themselves stay on disk).

Usage:
    uvicorn src.api.main:app --reload
    open http://127.0.0.1:8000/docs
"""

import json
import threading
import uuid
from datetime import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.add_app import COUNTRY_LANGS, LANGS, NAME, add_app, parse_link, read_registry, remove_app
from src.preprocessing.language import detect_lang
from src.sentiment.predict import predict
from src.topics.app_topics import OUT_DIR

SUMMARY_KEYS = ["app", "title", "period_days", "period_end", "first_day", "markets", "reviews", "negative_share",
                "complaints", "other_share", "several_share", "trend", "llm_model"]
PROBLEM_KEYS = ["id", "title", "description", "complaints", "share", "direction", "trend", "quotes", "related"]

app = FastAPI(title="Review Insights API",
              description="Тональність відгуків з Google Play, проблеми у скаргах і звіти для продуктової команди")
JOBS = {}
ADD_LOCK = threading.Lock()


class TextsIn(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=1000, examples=[["Після оновлення не працює оплата"]])


class AddIn(BaseModel):
    link: str = Field(examples=["https://play.google.com/store/apps/details?id=com.ubercab"])
    name: str | None = None
    country: str = Field("ua", examples=list(COUNTRY_LANGS))
    langs: list[str] | None = Field(None, description="Порожньо, щоб узяти типові мови для країни")
    per_lang: int = Field(1000, ge=100, le=3000)
    use_llm: bool = True


def now():
    return datetime.now().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check_name(name):
    if not NAME.match(name):
        raise HTTPException(400, "Назва застосунку: малі латинські літери, цифри або _")


def load_described(name):
    check_name(name)
    result = read_json(OUT_DIR / f"{name}_described.json")
    if result is None:
        raise HTTPException(404, f"Для {name} ще немає знайдених проблем")
    return result


def app_summary(name, registry):
    info = registry.get(name, {})
    result = read_json(OUT_DIR / f"{name}_described.json") or {}
    return {
        "name": name,
        "title": info.get("title") or result.get("title") or name,
        "added": name in registry,
        "period_end": result.get("period_end"),
        "complaints": result.get("complaints"),
        "problems": len(result["problems"]) if result else None,
        "report": (OUT_DIR / f"{name}_report.md").exists(),
        "llm_error": info.get("llm_error"),
    }


def run_job(job_id, request):
    job = JOBS[job_id]
    with ADD_LOCK:
        job.update(status="running", started=now())
        try:
            job["result"] = add_app(**request, log=job["log"].append)
            job["status"] = "done"
        except Exception as e:
            job.update(status="error", error=str(e))
        job["finished"] = now()

def active_job():
    return next((j for j in JOBS.values() if j["status"] in ("queued", "running")), None)

def start_job(request, background_tasks):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"id": job_id, "status": "queued", "created": now(), "request": request,
                    "log": [], "result": None, "error": None}
    background_tasks.add_task(run_job, job_id, request)
    return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}


@app.get("/health", summary="Перевірка, що сервер працює")
def health():
    return {"status": "ok"}


@app.post("/sentiment", summary="Тональність і мова для довільних текстів")
def sentiment(body: TextsIn):
    labels, confidence, model = predict(body.texts)
    return {
        "model": model,
        "results": [{"text": text, "sentiment": label, "confidence": round(float(c), 3), "lang": detect_lang(text)}
                    for text, label, c in zip(body.texts, labels, confidence)],
    }


@app.get("/apps", summary="Застосунки, для яких є дані")
def list_apps():
    registry = read_registry()
    names = {p.name.removesuffix("_described.json") for p in OUT_DIR.glob("*_described.json")} | set(registry)
    return [app_summary(name, registry) for name in sorted(names)]


@app.get("/apps/{name}/problems", summary="Проблеми у скаргах на застосунок")
def problems(name: str):
    result = load_described(name)
    summary = {k: result.get(k) for k in SUMMARY_KEYS}
    summary["title"] = summary["title"] or name
    summary["problems"] = [{k: p.get(k) for k in PROBLEM_KEYS} for p in result["problems"]]
    return summary


@app.get("/apps/{name}/problems/{problem_id}", summary="Одна проблема разом з усіма її скаргами")
def problem(name: str, problem_id: int, limit: int = 100):
    result = load_described(name)
    found = next((p for p in result["problems"] if p["id"] == problem_id), None)
    if found is None:
        raise HTTPException(404, f"У {name} немає проблеми {problem_id}")
    mine = [{"date": i["at"], "text": i["text"]} for i in result["items"]
            if i["current"] and problem_id in (i["problems"] or [])]
    mine.sort(key=lambda i: i["date"], reverse=True)
    return {**{k: found.get(k) for k in PROBLEM_KEYS}, "complaints_list": mine[:limit]}


@app.get("/apps/{name}/report", summary="Звіт для продуктової команди")
def report(name: str):
    check_name(name)
    markdown_path = OUT_DIR / f"{name}_report.md"
    if not markdown_path.exists():
        raise HTTPException(404, f"Для {name} ще немає звіту")
    return {"markdown": markdown_path.read_text(encoding="utf-8"), "report": read_json(OUT_DIR / f"{name}_report.json")}


@app.post("/apps", status_code=202, summary="Додати застосунок за посиланням з Google Play")
def add(body: AddIn, background_tasks: BackgroundTasks):
    try:
        parse_link(body.link)
    except ValueError as e:
        raise HTTPException(400, str(e))
    country = body.country.strip().lower()
    langs = body.langs or COUNTRY_LANGS.get(country, ["en"])
    if set(langs) - set(LANGS):
        raise HTTPException(400, f"Мови можна обирати лише з {', '.join(LANGS)}")
    request = {"link": body.link, "name": body.name, "country": country, "langs": langs,
               "per_lang": body.per_lang, "use_llm": body.use_llm}
    return start_job(request, background_tasks)


@app.post("/apps/{name}/update", status_code=202, summary="Дозібрати нові відгуки й оновити звіт")
def update(name: str, background_tasks: BackgroundTasks):
    info = read_registry().get(name)
    if info is None:
        raise HTTPException(404, f"{name} немає серед доданих застосунків")
    request = {"link": info["link"], "name": name, "country": info["country"], "langs": info["languages"],
               "per_lang": info["per_lang"], "use_llm": True}
    return start_job(request, background_tasks)


@app.delete("/apps/{name}", summary="Видалити доданий застосунок")
def delete(name: str):
    check_name(name)
    busy = active_job()
    if busy:
        raise HTTPException(409, f"Зараз виконується завдання {busy['id']}. Видалити можна після його завершення")
    try:
        info = remove_app(name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"removed": name, "title": info["title"]}


@app.get("/jobs/{job_id}", summary="Стан додавання чи оновлення застосунку")
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Такого завдання немає. Після перезапуску сервера завдання зникають")
    return JOBS[job_id]