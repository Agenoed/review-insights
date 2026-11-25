"""Add a new app by its Google Play link: collect reviews, clean them, score sentiment,
find complaint problems and write the report.

Added apps are kept apart from the main dataset in data/added, so the train/val/test
split and the metrics of the models stay the same. Running it again for the same app
adds only new reviews and builds the problems and the report again.

Usage:
    python -m src.pipeline.add_app "https://play.google.com/store/apps/details?id=com.ubercab"
    python -m src.pipeline.add_app com.ubercab --name uber --country pl --langs pl en --per-lang 1000
    python -m src.pipeline.add_app com.ubercab --no-llm
    python -m src.pipeline.add_app --remove uber
"""

import argparse
import json
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pandas as pd
from google_play_scraper import app as play_app

from src.data.collect_google_play import collect
from src.llm.report import build_report
from src.llm.topics import describe_app
from src.preprocessing.language import detect_lang
from src.preprocessing.prepare import PROCESSED_PATH, SENTIMENT, clean_text
from src.sentiment.predict import predict
from src.topics.app_topics import ADDED_DIR, OUT_DIR, ROOT, load_reviews

REGISTRY_PATH = ADDED_DIR / "apps.json"
RAW_ADDED_DIR = ADDED_DIR / "raw"
LANGS = ["uk", "ru", "en", "pl", "es"]
COUNTRY_LANGS = {"ua": ["uk", "ru", "en"], "pl": ["pl", "en"], "es": ["es", "en"], "mx": ["es", "en"],
                 "us": ["en"], "gb": ["en"]}
APP_ID = re.compile(r"^[A-Za-z][\w]*(\.[\w]+)+$")
NAME = re.compile(r"^[a-z0-9_]{2,30}$")
COUNTRY = re.compile(r"^[a-z]{2}$")
TITLE_END = re.compile(r"\s[-–—]\s|[:|(,]")
TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie", "ж": "zh", "з": "z",
    "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ь": "", "ю": "iu", "я": "ia", "ы": "y", "э": "e", "ъ": "", "ё": "e",
})


def parse_link(link):
    link = link.strip()
    app_id = parse_qs(urlparse(link).query).get("id", [link])[0].strip()
    if not APP_ID.match(app_id):
        raise ValueError("Це не схоже на посилання Google Play або id застосунку, наприклад com.glovo")
    return app_id


def default_name(title, app_id):
    main = TITLE_END.split(title, maxsplit=1)[0]
    words = [re.sub(r"[^a-z0-9]", "", w.lower().translate(TRANSLIT)) for w in main.split()]
    name = "_".join(w for w in words if w)[:30].rstrip("_")
    if len(name) < 2:
        name = re.sub(r"[^a-z0-9_]", "", app_id.split(".")[-1].lower())[:30]
    return name

def clean_name(name):
    name = re.sub(r"[\s-]+", "_", name.strip().lower().translate(TRANSLIT))
    return re.sub(r"[^a-z0-9_]", "", name)

def main_apps():
    names, ids = set(), {}
    for path in (ROOT / "configs").glob("apps*.json"):
        for a in json.loads(path.read_text(encoding="utf-8")).get("apps", []):
            names.add(a["name"])
            ids[a["id"]] = a["name"]
    if PROCESSED_PATH.exists():
        names |= set(pd.read_csv(PROCESSED_PATH, usecols=["app"])["app"].unique())
    return names, ids


def read_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def save_registry(registry):
    ADDED_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def check(app_id, name, country, langs):
    if not NAME.match(name):
        raise ValueError("Назва має бути від 2 до 30 символів: малі латинські літери, цифри або _")
    if not COUNTRY.match(country):
        raise ValueError("Країна має бути двобуквеним кодом, наприклад ua, pl, es")
    if not langs or set(langs) - set(LANGS):
        raise ValueError(f"Мови можна обирати лише з {', '.join(LANGS)}")
    names, ids = main_apps()
    if app_id in ids:
        raise ValueError(f"Цей застосунок уже є в основному наборі даних як {ids[app_id]}")
    if name in names:
        raise ValueError(f"Назва {name} уже зайнята застосунком з основного набору")
    for other, info in read_registry().items():
        if info["id"] == app_id and other != name:
            raise ValueError(f"Цей застосунок уже додано під назвою {other}")
        if other == name and info["id"] != app_id:
            raise ValueError(f"Назва {name} уже зайнята іншим доданим застосунком")


def prepare_rows(raw):
    df = raw.copy()
    df["at"] = pd.to_datetime(df["at"])
    df["text"] = df["content"].map(clean_text)
    df["lang"] = df["text"].map(detect_lang)
    df = df[df["lang"] != "no_letters"].copy()
    df["sentiment"] = df["score"].map(SENTIMENT)
    df["split"] = "added"
    return df.reset_index(drop=True)


def merge_and_score(name, rows):
    path = ADDED_DIR / f"{name}.csv"
    old = pd.read_csv(path, parse_dates=["at"], keep_default_na=False) if path.exists() else None
    new = rows if old is None else rows[~rows["reviewId"].isin(old["reviewId"])]
    if len(new):
        labels, confidence, model = predict(new["text"])
        new = new.assign(pred=labels, confidence=confidence.round(3), model=model)
    df = new if old is None else pd.concat([old, new], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return df, len(new)

def remove_app(name):
    registry = read_registry()
    if name not in registry:
        raise ValueError(f"{name} немає серед доданих застосунків. Видаляти можна лише їх")
    paths = [ADDED_DIR / f"{name}.csv", *RAW_ADDED_DIR.glob(f"{name}_[0-9]*.csv"),
             *(OUT_DIR / f"{name}{end}" for end in ("_described.json", "_report.json", "_report.md"))]
    for path in paths:
        path.unlink(missing_ok=True)
    info = registry.pop(name)
    save_registry(registry)
    return info

def add_app(link, name=None, country="ua", langs=("uk", "ru", "en"), per_lang=1000, provider=None,
            use_llm=True, pause=1.0, log=print):
    start = time.perf_counter()
    app_id = parse_link(link)
    langs = list(langs)
    try:
        info = play_app(app_id, lang=langs[0] if langs else "en", country=country)
    except Exception as e:
        raise ValueError(f"Не вдалося знайти {app_id} у Google Play ({country}): {e}") from e
    title = info.get("title") or app_id
    name = clean_name(name) if name else default_name(title, app_id)
    check(app_id, name, country, langs)
    log(f"{title} ({app_id}), назва в системі: {name}")

    category = (info.get("genreId") or "unknown").lower()
    config = {"country": country, "languages": langs, "apps": [{"name": name, "id": app_id, "category": category}]}
    raw = collect(config, per_lang, pause)
    if raw.empty:
        raise ValueError("Google Play не віддав жодного відгуку для цих країни й мов")
    RAW_ADDED_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_ADDED_DIR / f"{name}_{datetime.now():%Y%m%d_%H%M}.csv", index=False, encoding="utf-8-sig")
    log(f"Зібрано {len(raw)} відгуків")

    df, added = merge_and_score(name, prepare_rows(raw))
    negative = float((df["pred"] == "negative").mean())
    log(f"Нових відгуків: {added}, разом {len(df)}, негативних {negative:.0%}")

    registry = read_registry()
    entry = registry.get(name, {"added": datetime.now().isoformat(timespec="seconds")})
    entry.update({
        "id": app_id, "title": title, "link": f"https://play.google.com/store/apps/details?id={app_id}",
        "category": category, "country": country, "languages": langs, "per_lang": per_lang,
        "updated": datetime.now().isoformat(timespec="seconds"), "reviews": len(df),
        "negative_share": round(negative, 3),
    })
    registry[name] = entry
    save_registry(registry)

    if use_llm:
        try:
            log("Шукаю теми скарг і проблеми")
            result = describe_app(name, provider=provider, df=load_reviews())
            result["title"] = title
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / f"{name}_described.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"Знайдено проблем: {len(result['problems'])}. Пишу звіт")
            report, markdown = build_report(name, provider)
            (OUT_DIR / f"{name}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            (OUT_DIR / f"{name}_report.md").write_text(markdown, encoding="utf-8")
            entry.update({"problems": len(result["problems"]), "complaints": result["complaints"],
                          "report_updated": datetime.now().isoformat(timespec="seconds"), "llm_error": None})
        except Exception as e:
            entry["llm_error"] = "замало скарг для пошуку проблем" if "too few complaints" in str(e) else str(e)
            log(f"Відгуки й тональність збережено, але проблеми чи звіт не вийшли: {entry['llm_error']}")
        save_registry(registry)

    entry["seconds"] = round(time.perf_counter() - start)
    log(f"Готово за {entry['seconds']} с")
    return {"name": name, **entry}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("link", nargs="?")
    parser.add_argument("--remove", metavar="NAME")
    parser.add_argument("--name")
    parser.add_argument("--country", default="ua")
    parser.add_argument("--langs", nargs="+", default=["uk", "ru", "en"])
    parser.add_argument("--per-lang", type=int, default=1000)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    if args.remove:
        try:
            info = remove_app(args.remove)
        except ValueError as e:
            raise SystemExit(str(e))
        print(f"Removed {args.remove} ({info['title']})")
        return
    if not args.link:
        parser.error("give a Google Play link or --remove NAME")
    try:
        result = add_app(args.link, args.name, args.country, args.langs, args.per_lang, args.provider, not args.no_llm)
    except ValueError as e:
        raise SystemExit(str(e))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()