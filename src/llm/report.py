"""Report for the product team of one app, built from the problems found by
src.llm.topics. For every main problem the LLM reads its complaints and writes
concrete details with the numbers of the complaints that mention them, then
writes a summary and recommendations and translates the quotes. All counts and
shares are put in by code, links between problems are computed from complaints
that mention several problems.

Usage:
    python -m src.llm.report --app glovo
    python -m src.llm.report --app diia --provider ollama
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from src.llm.client import get_model, get_provider
from src.llm.topics import (
    RUSSIAN_ONLY, UKRAINIAN_ONLY, WORKERS, ask, foreign_letters, interleave, numbered, to_index, to_indices,
)
from src.topics.app_topics import OUT_DIR
from src.topics.modeling import shorten

DETAILED = 8
READ_COMPLAINTS = 15
MIN_DETAIL_COMPLAINTS = 2
MIN_LINK_COMPLAINTS = 5
MIN_LINK_SHARE = 0.25
MAX_LINKS = 5
MIN_INFORMATIVE_CHARS = 80
MARKDOWN_CHARS = re.compile(r"([\\`*_#>|\[\]])")
NUMBER_LIST = re.compile(r"\s*[\[(]\s*\d+(?:\s*,\s*\d+)*\s*[\])]")

DETAILS_PROMPT = """You help the product team of the app "{app}" understand one problem from app store complaints.
Problem: {title}. {description}
The user message contains {n} numbered complaints about this problem. They can be in different languages.
Write in Ukrainian:
- details: up to 5 concrete details of this problem that repeat in the complaints and help to reproduce or fix it, for example on which step it happens, the error message, devices or versions, since when it happens, what users already tried. Each detail is a short phrase without complaint numbers in it, and the numbers of the complaints that mention it go to "complaints". Complaints often mention other problems too, for example support or refunds: skip them, write only about this problem. Use only what is written in the complaints
- quotes: numbers of the 2 complaints that show the problem best and are clear for a manager
Write only in Ukrainian; names of apps, services, devices and error messages can stay as they are.
Return only JSON: {{"details": [{{"text": "...", "complaints": [1, 4]}}], "quotes": [1, 2]}}"""

REPORT_PROMPT = """You write a short report for the product team of the app "{app}" about what users complain about in app store reviews.
The user message contains facts computed by the system for the last {days} days: overall numbers and the problems found in complaints, numbered, with how many complaints mention each problem, whether it is growing compared with the previous period and concrete details from the complaints.
Write in Ukrainian:
- summary: 2-3 sentences: what bothers users most and what changed. Do not write any numbers, the system adds them
- actions: 3-5 recommendations for the product team, the most important first. Each one is about one problem: its number and one sentence about what exactly to check or fix, based on its details. Do not repeat the name of the problem in the sentence and do not write percentages
Use only the given facts, do not invent causes or details.
Return only JSON: {{"summary": "...", "actions": [{{"problem": 1, "text": "..."}}]}}"""

TRANSLATE_PROMPT = """Translate every numbered app store review from the user message into Ukrainian, including reviews in Russian.
If a review is already in Ukrainian, return it unchanged. Keep the meaning and the tone, do not shorten and do not add anything. Keep names of apps, shops and services as they are.
Return only JSON with every review number as a key: {"1": "...", "2": "..."}"""


def complaints_word(n):
    if n % 10 == 1 and n % 100 != 11:
        return "скарга"
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return "скарги"
    return "скарг"


def percent(value):
    return f"{value * 100:.0f}%"


def is_ukrainian(text):
    return any(ch in UKRAINIAN_ONLY for ch in text) and not any(ch in RUSSIAN_ONLY for ch in text)


def lower_first(text):
    return text[0].lower() + text[1:] if len(text) > 1 and not text[1].isupper() else text


def escape(text):
    return MARKDOWN_CHARS.sub(r"\\\1", text)


def trend_text(problem):
    trend = problem["trend"]
    if not trend:
        return ""
    before, now = percent(trend["previous_share"]), percent(trend["share"])
    if problem["direction"] == "growing":
        return f"Росте: з {before} до {now} скарг."
    if problem["direction"] == "declining":
        return f"Меншає: з {before} до {now} скарг."
    return f"Було {before}, зараз {now} скарг, різниця статистично незначуща."


def complaints_to_read(problem, items, clusters):
    mine = [i for i in items if i["current"] and i["problems"] and problem["id"] in i["problems"]]
    by_text = {shorten(i["text"]): i for i in mine}
    typical = interleave([c["examples"] for c in clusters if c["topic"] in problem["groups"]], 100)
    ordered = list(dict.fromkeys([t for t in typical if t in by_text] + list(by_text)))
    ordered.sort(key=lambda t: (len(t) < MIN_INFORMATIVE_CHARS, len(by_text[t]["problems"]) > 1))
    return [by_text[t] for t in ordered[:READ_COMPLAINTS]]


def good_details(answer):
    details = answer.get("details")
    if not isinstance(details, list) or not all(isinstance(d, dict) and isinstance(d.get("text"), str) for d in details):
        return False
    joined = " ".join(d["text"] for d in details)
    return not foreign_letters(joined) and (not details or any(ch in UKRAINIAN_ONLY for ch in joined))


def read_problem(app, problem, complaints, provider):
    texts = [shorten(i["text"]) for i in complaints]
    system = DETAILS_PROMPT.format(app=app, title=problem["title"], description=problem["description"], n=len(texts))
    answer = ask(system, numbered(texts), provider, check=good_details) or {}
    details = []
    for d in answer.get("details") or []:
        if isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"].strip():
            support = to_indices(d.get("complaints"), len(texts))
            if len(support) >= MIN_DETAIL_COMPLAINTS:
                            details.append({"text": NUMBER_LIST.sub("", d["text"]).strip(), "complaints": len(support),
                                "ids": [complaints[k - 1]["id"] for k in support]})
    picked = to_indices(answer.get("quotes"), len(texts))
    picked += [i for i in range(1, len(texts) + 1) if i not in picked]
    return {"details": details, "read": len(texts), "quotes": [texts[i - 1] for i in picked[:2]]}


def facts(result, problems, read):
    lines = [f"Reviews: {result['reviews']}, negative: {percent(result['negative_share'])}."]
    if result["trend"]:
        lines.append(f"Negative reviews in the previous period: {percent(result['trend']['previous_negative_share'])}.")
    lines.append(f"Complaints: {result['complaints']}, without any of the problems: {percent(result['other_share'])}.")
    lines.append("Problems:")
    for k, p in enumerate(problems, 1):
        line = (f"{k}. {p['title']}. {p['description']} Complaints: {p['complaints']} ({percent(p['share'])}). "
                f"Change: {p['direction'] or 'unknown'}.")
        details = read.get(p["id"], {}).get("details")
        if details:
            line += " Details: " + "; ".join(d["text"] for d in details) + "."
        lines.append(line)
    return "\n".join(lines)


def good_texts(answer, n):
    summary, actions = answer.get("summary"), answer.get("actions")
    if not (isinstance(summary, str) and summary.strip() and isinstance(actions, list) and actions):
        return False
    for a in actions:
        if not (isinstance(a, dict) and to_index(a.get("problem"), n) and isinstance(a.get("text"), str) and a["text"].strip()):
            return False
    joined = " ".join([summary] + [a["text"] for a in actions])
    return (not re.search(r"\d", summary) and "%" not in joined and not foreign_letters(joined)
            and any(ch in UKRAINIAN_ONLY for ch in joined))


def write_texts(app, result, problems, read, provider):
    system = REPORT_PROMPT.format(app=app, days=result["period_days"])
    answer = ask(system, facts(result, problems, read), provider, check=lambda a: good_texts(a, len(problems))) or {}
    if not good_texts(answer, len(problems)):
        print("  the report text did not pass the check, check it by hand")
    actions = []
    for a in answer.get("actions") or []:
        k = to_index(a.get("problem"), len(problems)) if isinstance(a, dict) else None
        if k and isinstance(a.get("text"), str) and a["text"].strip():
            actions.append({"problem": problems[k - 1]["id"], "text": a["text"].strip()})
    summary = answer.get("summary")
    return {"summary": summary.strip() if isinstance(summary, str) else "", "actions": actions}


def translate(quotes, provider):
    todo = [q for q in dict.fromkeys(quotes) if not is_ukrainian(q)]
    if not todo:
        return {}
    keys = [str(i) for i in range(1, len(todo) + 1)]

    def good(answer):
        return all(isinstance(answer.get(k), str) and (answer[k].strip() == q.strip() or is_ukrainian(answer[k]))
                   for k, q in zip(keys, todo))

    answer = ask(TRANSLATE_PROMPT, numbered(todo), provider, check=good) or {}
    result = {}
    for k, quote in zip(keys, todo):
        text = answer.get(k)
        if isinstance(text, str) and text.strip() != quote.strip() and is_ukrainian(text) and not foreign_letters(text):
            result[quote] = text.strip()
    return result


def links(problems):
    by_id = {p["id"]: p for p in problems}
    found = {}
    for p in problems:
        for r in p["related"]:
            q = by_id.get(r["id"])
            if not q:
                continue
            small, big = (p, q) if p["complaints"] <= q["complaints"] else (q, p)
            share = r["complaints"] / small["complaints"]
            if r["complaints"] >= MIN_LINK_COMPLAINTS and share >= MIN_LINK_SHARE:
                found[(small["id"], big["id"])] = (share, r["complaints"], small, big)
    best = sorted(found.values(), key=lambda x: -x[0])[:MAX_LINKS]
    return [f"{percent(share)} скарг про «{small['title']}» ({both} з {small['complaints']}) також стосуються "
            f"проблеми «{big['title']}»." for share, both, small, big in best]


def render(result, problems, read, texts, translations, report_model):
    titles = {p["id"]: p["title"] for p in problems}
    lines = [f"# Скарги користувачів: {result['app']}", ""]
    period = (f"Період: останні {result['period_days']} днів, до {result['period_end']}. "
              f"Відгуків за період: {result['reviews']}, негативних {percent(result['negative_share'])}")
    if result["trend"]:
        period += f" (у попередньому періоді {percent(result['trend']['previous_negative_share'])})"
    lines += [period + f". Скарг, які розібрала система: {result['complaints']}.", ""]

    if texts["summary"]:
        lines += ["## Коротко", "", texts["summary"], ""]

    lines += ["## Головні проблеми", ""]
    for k, p in enumerate(problems[:DETAILED], 1):
        lines += [f"### {k}. {p['title']}", ""]
        lines += [f"{p['complaints']} {complaints_word(p['complaints'])}, {percent(p['share'])} усіх скарг. {trend_text(p)}".strip(), ""]
        lines += [p["description"], ""]
        info = read.get(p["id"], {})
        if info.get("details"):
            lines += [f"Що саме пишуть користувачі (у дужках: скільки з {info['read']} переглянутих скарг про це згадують):", ""]
            lines += [f"- {escape(d['text'])} ({d['complaints']})" for d in info["details"]]
            lines.append("")
        elif info:
            lines += [f"Конкретних деталей, які повторюються хоча б у {MIN_DETAIL_COMPLAINTS} з {info['read']} переглянутих скарг, немає.", ""]
        for quote in info.get("quotes") or p["quotes"]:
            lines.append(f"> {escape(quote)}")
            if quote in translations:
                lines += [">", f"> *Переклад:* {escape(translations[quote])}"]
            lines.append("")

    rare = problems[DETAILED:]
    if rare:
        lines += ["## Рідші проблеми", ""]
        lines += [f"- {p['title']}: {p['complaints']} {complaints_word(p['complaints'])}. {p['description']}" for p in rare]
        lines.append("")

    link_lines = links(problems)
    if link_lines:
        lines += ["## Які проблеми йдуть разом", ""]
        lines += [f"- {line}" for line in link_lines]
        lines.append("")

    if texts["actions"]:
        lines += ["## Що варто зробити", ""]
        lines += [f"{k}. «{titles[a['problem']]}»: {lower_first(a['text'])}" for k, a in enumerate(texts["actions"], 1)]
        lines.append("")

    if report_model == result["llm_model"]:
        models = f"Список проблем, віднесення скарг до них, текст звіту й переклади цитат зробила мовна модель {report_model}. "
    else:
        models = (f"Список проблем і віднесення скарг до них зробила мовна модель {result['llm_model']}, "
                  f"текст звіту й переклади цитат {report_model}. ")
    lines += [
        "## Як рахували", "",
        f"До жодної з проблем не віднесено {percent(result['other_share'])} скарг: це загальні оцінки "
        "на кшталт «жахливий додаток» або поодинокі випадки. Одна скарга може описувати кілька проблем "
        f"(таких {percent(result['several_share'])}), тому частки в сумі дають більше 100%.",
        "",
        "Скаргами вважаються відгуки від 30 символів, які модель тональності віднесла до негативних. " + models +
        "Кількості, частки й тренди пораховані кодом. Тренд позначено як ріст чи спад, лише якщо зміна частки "
        "статистично значуща (z-тест для двох часток, рівень 0,05). Деталі до кожної проблеми модель виписала "
        f"з її найтиповіших скарг (до {READ_COMPLAINTS}), а в звіт потрапили лише ті, що згадані щонайменше в "
        f"{MIN_DETAIL_COMPLAINTS} з них.",
    ]
    return "\n".join(lines) + "\n"


def build_report(app, provider=None):
    provider = get_provider(provider)
    result = json.loads((OUT_DIR / f"{app}_described.json").read_text(encoding="utf-8"))
    problems, items, clusters = result["problems"], result["items"], result["clusters"]

    top = problems[:DETAILED]
    to_read = [complaints_to_read(p, items, clusters) for p in top]
    with ThreadPoolExecutor(WORKERS) as pool:
        found = list(pool.map(lambda pt: read_problem(app, pt[0], pt[1], provider), zip(top, to_read)))
    read = {p["id"]: info for p, info in zip(top, found)}
    for p in top:
        print(f"  read: {p['title']} ({len(read[p['id']]['details'])} details)")

    texts = write_texts(app, result, problems, read, provider)
    translations = translate([q for info in read.values() for q in info["quotes"]], provider)
    report = {
        "app": app,
        "period_days": result["period_days"],
        "period_end": result["period_end"],
        "problems": {str(pid): info for pid, info in read.items()},
        "links": links(problems),
        "texts": texts,
        "translations": translations,
        "llm_provider": provider,
        "llm_model": get_model(provider),
    }
    return report, render(result, problems, read, texts, translations, report["llm_model"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    args = parser.parse_args()

    report, markdown = build_report(args.app, args.provider)
    (OUT_DIR / f"{args.app}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    out = OUT_DIR / f"{args.app}_report.md"
    out.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()