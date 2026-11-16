"""Problems users complain about in one app. The LLM gets the most typical complaints
of every cluster and writes one list of distinct problems, then sorts every
complaint of the period by these problems (one complaint can have several problems
or none). Complaints left without a problem are read once more to find problems
the first list missed. Problems that share most of their complaints are checked
and merged.
The LLM only writes text and picks numbers; all counts, shares and trends are
computed in code from the sorted complaints.

Usage:
    python -m src.llm.topics --app glovo
    python -m src.llm.topics --app diia --provider ollama
"""

import argparse
import itertools
import json
import math
import random
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src.llm.client import chat, get_model, get_provider
from src.topics.app_topics import OUT_DIR, find_app_topics, load_reviews, share
from src.topics.modeling import shorten

MAX_TOPICS = 15
MIN_TOPIC_COMPLAINTS = 3
MIN_GROUP_EXAMPLES = 5
MAX_GROUP_EXAMPLES = 15
MAX_PROBLEMS = 15
MIN_PROBLEM_COMPLAINTS = 3
MIN_LEFT_FOR_MORE = 15
MAX_LEFT_EXAMPLES = 100
OVERLAP_LIMIT = 0.5
ASSIGN_BATCH = 20
ASSIGN_CHARS = 500
WORKERS = 4
RUSSIAN_ONLY = set("ыэъёЫЭЪЁ")
UKRAINIAN_ONLY = set("іїєґІЇЄҐ")

PROBLEM_RULES = """- title: one problem in 3-8 words, as a concrete situation a developer can reproduce or a product manager can act on. Start with what exactly happens, for example "Не синхронізуються кроки з годинником", "Книга відкривається з початку замість закладки", "Зникає звук у відео після блокування екрана". Do not join two different problems with "та" or "або", do not start with general words like "Проблеми з", "Помилки при", "Неможливість"
- description: 1-2 sentences: what exactly happens, when, and what users expect. Use only facts from the complaints, do not add anything"""

DISCOVER_PROMPT = """You analyze complaints from app store reviews of the app "{app}" for its product team.
The user message contains groups of similar complaints, each with its size and its most typical complaints. They can be in different languages.
Write the list of distinct problems users complain about, in Ukrainian:
{rules}
- groups: numbers of the groups where this problem appears
Rules:
- Problems must not overlap. If several groups describe the same problem, for example in different languages, write it once. If one group mixes different problems, write them as separate problems
- Write a problem only if at least 3 complaints describe it. Skip complaints that are too general to tell, like "terrible app"
- Order the problems from the most common to the least common
Write only in Ukrainian; names of apps, banks, services and functions can stay as they are.
Return only JSON: {{"problems": [{{"title": "...", "description": "...", "groups": [1, 2]}}]}}"""

MORE_PROMPT = """You analyze complaints from app store reviews of the app "{app}" for its product team.
Titles of the problems already found:
{problems}
The user message contains complaints that were not sorted into any of these problems. They can be in different languages.
Write the new problems from these complaints, in Ukrainian:
{rules}
Rules:
- Do not repeat or rephrase the problems from the list. A problem counts as found only if its title is in the list, even if it is often mentioned together with one of them
- Write a problem only if at least 3 of these complaints describe it. Skip complaints that are too general to tell, like "terrible app", and complaints without any problem
- If there are no such problems, return an empty list
Write only in Ukrainian; names of apps, banks, services and functions can stay as they are.
Return only JSON: {{"problems": [{{"title": "...", "description": "..."}}]}}"""

ASSIGN_PROMPT = """You sort complaints from app store reviews of the app "{app}" by problem.
Problems, each with a title and details:
{problems}
The user message contains {n} numbered complaints. They can be in different languages.
For every complaint list the numbers of the problems that are clearly present in it. One complaint can have several problems or none. Assign a problem only if the complaint describes the problem from its title, not just something from the same area. Complaints that are too general to tell, like "terrible app", get no problems.
Return only JSON with every complaint number as a key, for example: {{"1": [2], "2": [], "3": [1, 4]}}"""

SAME_PROMPT = """You check a list of problems from app store reviews of the app "{app}".
The user message contains two problems and how many complaints were sorted into both of them.
Decide whether they are the same problem written in other words. Answer false if they are different problems that often happen together, like a delayed train and a missed connection.
Return only JSON: {{"same": true}} or {{"same": false}}"""


def parse_object(answer):
    start, end = answer.find("{"), answer.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON in the answer")
    return json.loads(answer[start:end + 1])


def ask(system, user, provider, check=None, retries=3):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    last = None
    for attempt in range(retries):
        try:
            answer = parse_object(chat(messages, provider=provider, temperature=0 if attempt == 0 else 0.3))
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(10 * (attempt + 1))
            continue
        if check is None or check(answer):
            return answer
        last = answer
        print(f"  attempt {attempt + 1}: the answer did not pass the check")
    return last


def to_index(value, n):
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return i if 1 <= i <= n else None


def to_indices(values, n):
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(i for i in (to_index(v, n) for v in values) if i))


def letters(text):
    return [ch for ch in text if unicodedata.category(ch) in ("Lu", "Ll", "Lo")]


def foreign_letters(text):
    return any(ch in RUSSIAN_ONLY or not unicodedata.name(ch, "").startswith(("CYRILLIC", "LATIN"))
               for ch in letters(text))


def good_text(answer):
    title, description = answer.get("title"), answer.get("description")
    if not (isinstance(title, str) and isinstance(description, str) and title.strip() and description.strip()):
        return False
    return not foreign_letters(title + " " + description) and any(ch in UKRAINIAN_ONLY for ch in description)


def numbered(texts):
    return "\n".join(f"{i}. {text}" for i, text in enumerate(texts, 1))


def interleave(lists, limit):
    result = []
    for i in range(max(map(len, lists), default=0)):
        for items in lists:
            if i < len(items) and len(result) < limit:
                result.append(items[i])
    return result


def good_problems(answer, allow_empty=False):
    problems = answer.get("problems")
    if not isinstance(problems, list) or not (problems or allow_empty):
        return False
    return all(isinstance(p, dict) and good_text(p) for p in problems)


def read_problems(answer, first_id, clusters=None):
    found = answer.get("problems") if isinstance(answer.get("problems"), list) else []
    problems = []
    for p in found:
        if isinstance(p, dict) and good_text(p):
            groups = to_indices(p.get("groups"), len(clusters)) if clusters else []
            problems.append({
                "id": first_id + len(problems),
                "title": p["title"].strip(),
                "description": p["description"].strip(),
                "groups": [clusters[k - 1]["topic"] for k in groups],
            })
    return problems


def discover(app, clusters, provider):
    blocks = []
    for k, cluster in enumerate(clusters, 1):
        n = min(MAX_GROUP_EXAMPLES, max(MIN_GROUP_EXAMPLES, cluster["complaints"] // 4))
        lines = "\n".join(f"- {text}" for text in cluster["examples"][:n])
        blocks.append(f"Group {k} ({cluster['complaints']} complaints):\n{lines}")
    system = DISCOVER_PROMPT.format(app=app, rules=PROBLEM_RULES)
    answer = ask(system, "\n\n".join(blocks), provider, check=good_problems) or {}
    problems = read_problems(answer, 1, clusters)[:MAX_PROBLEMS]
    if not problems:
        raise ValueError("the LLM did not return any problems")
    return problems


def discover_more(app, problems, items, provider):
    left = [i for i in items if i["current"] and not i["problems"]]
    if len(left) < MIN_LEFT_FOR_MORE or len(problems) >= MAX_PROBLEMS:
        return []
    random.Random(42).shuffle(left)
    listing = "\n".join(f"- {p['title']}" for p in problems)
    system = MORE_PROMPT.format(app=app, problems=listing, rules=PROBLEM_RULES)
    user = "\n".join(f"- {shorten(i['text'])}" for i in left[:MAX_LEFT_EXAMPLES])
    answer = ask(system, user, provider, check=lambda a: good_problems(a, allow_empty=True)) or {}
    first_id = max(p["id"] for p in problems) + 1
    return read_problems(answer, first_id)[:MAX_PROBLEMS - len(problems)]


def assign_batch(app, problems, texts, provider):
    listing = "\n".join(f"{i}. {p['title']}. {p['description']}" for i, p in enumerate(problems, 1))
    system = ASSIGN_PROMPT.format(app=app, problems=listing, n=len(texts))
    keys = [str(i) for i in range(1, len(texts) + 1)]
    answer = ask(system, numbered(texts), provider, check=lambda a: all(isinstance(a.get(k), list) for k in keys)) or {}
    missing = sum(not isinstance(answer.get(k), list) for k in keys)
    if missing:
        print(f"  no answer for {missing} complaints, they stay without problems")
    return [[problems[n - 1]["id"] for n in to_indices(answer.get(k), len(problems))] for k in keys]


def assign(app, problems, items, provider):
    todo = [item for item in items if item["current"] or item["in_trend"]]
    batches = [todo[start:start + ASSIGN_BATCH] for start in range(0, len(todo), ASSIGN_BATCH)]

    def run(batch):
        return assign_batch(app, problems, [shorten(item["text"], ASSIGN_CHARS) for item in batch], provider)

    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for batch, found in zip(batches, pool.map(run, batches)):
            for item, problem_ids in zip(batch, found):
                item["problems"] = (item["problems"] or []) + problem_ids
            done += len(batch)
            print(f"  assigned {done}/{len(todo)}")


def is_same(app, a, b, both, provider):
    user = f"A: {a['title']}. {a['description']}\nB: {b['title']}. {b['description']}\nComplaints sorted into both: {both}"
    answer = ask(SAME_PROMPT.format(app=app), user, provider) or {}
    same = answer.get("same")
    if isinstance(same, str):
        same = same.strip().lower() == "true"
    return same is True


def refine(app, problems, items, provider):
    by_id = {p["id"]: p for p in problems}
    sets = {pid: {i["id"] for i in items if i["problems"] and pid in i["problems"]} for pid in by_id}
    pairs = []
    for a, b in itertools.combinations(by_id, 2):
        small, both = min(len(sets[a]), len(sets[b])), len(sets[a] & sets[b])
        if small >= MIN_PROBLEM_COMPLAINTS and both / small >= OVERLAP_LIMIT:
            pairs.append((both / small, both, a, b))

    alias, merged = {}, []
    for overlap, both, a, b in sorted(pairs, reverse=True):
        a, b = alias.get(a, a), alias.get(b, b)
        if a == b:
            continue
        keep, drop = (a, b) if len(sets[a]) >= len(sets[b]) else (b, a)
        if is_same(app, by_id[keep], by_id[drop], both, provider):
            for old, new in list(alias.items()):
                if new == drop:
                    alias[old] = keep
            alias[drop] = keep
            by_id[keep]["groups"] = list(dict.fromkeys(by_id[keep]["groups"] + by_id[drop]["groups"]))
            merged.append({"kept": by_id[keep]["title"], "merged": by_id[drop]["title"], "overlap": round(overlap, 3)})
            print(f"  merged «{by_id[drop]['title']}» into «{by_id[keep]['title']}» (overlap {overlap:.0%})")

    for item in items:
        if item["problems"] is not None:
            item["problems"] = list(dict.fromkeys(alias.get(pid, pid) for pid in item["problems"]))
    return [p for p in problems if p["id"] not in alias], merged


def trend_direction(trend, totals, z_limit=1.96):
    if not trend or not totals:
        return None
    x1, x0 = trend["complaints"], trend["previous_complaints"]
    n1, n0 = totals["complaints"], totals["previous_complaints"]
    if x1 + x0 < 10:
        return "stable"
    p = (x1 + x0) / (n1 + n0)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    z = (x1 / n1 - x0 / n0) / se if se else 0
    return "growing" if z > z_limit else "declining" if z < -z_limit else "stable"


def pick_quotes(problem, mine, clusters):
    texts = {shorten(i["text"]): i for i in mine}
    groups = [c["examples"] for c in clusters if c["topic"] in problem["groups"]]
    candidates = [text for text in interleave(groups, 100) if text in texts] + list(texts)
    pure = [text for text in candidates if len(texts[text]["problems"]) == 1]
    return list(dict.fromkeys(pure + candidates))[:2]


def count(problems, items, clusters, totals):
    current = [i for i in items if i["current"]]
    now_items = [i for i in current if i["in_trend"]]
    before_items = [i for i in items if not i["current"] and i["in_trend"]]
    for p in problems:
        pid = p["id"]
        mine = [i for i in current if pid in i["problems"]]
        p["complaints"] = len(mine)
        p["share"] = share(len(mine), len(current))
        p["trend"] = None
        if totals:
            now = sum(pid in i["problems"] for i in now_items)
            before = sum(pid in i["problems"] for i in before_items)
            p["trend"] = {
                "complaints": now,
                "previous_complaints": before,
                "share": share(now, totals["complaints"]),
                "previous_share": share(before, totals["previous_complaints"]),
            }
        p["direction"] = trend_direction(p["trend"], totals)
        p["quotes"] = pick_quotes(p, mine, clusters)
        together = Counter(other for i in mine for other in i["problems"] if other != pid)
        p["related"] = [{"id": k, "complaints": v} for k, v in together.most_common(3) if v >= 3]


def cluster_purity(items):
    groups = {}
    for item in items:
        if item["current"] and item["topic"] != -1:
            groups.setdefault(item["topic"], []).append(item)
    top = 0
    for group in groups.values():
        counts = Counter(pid for i in group for pid in i["problems"])
        top += counts.most_common(1)[0][1] if counts else 0
    return share(top, sum(map(len, groups.values())))


def describe_app(app, days=None, provider=None, df=None):
    provider = get_provider(provider)
    result = find_app_topics(load_reviews() if df is None else df, app, days)
    items, all_clusters = result.pop("items"), result.pop("topics")
    clusters = [c for c in all_clusters if c["complaints"] >= MIN_TOPIC_COMPLAINTS][:MAX_TOPICS]

    for item in items:
        item["problems"] = None
    problems = discover(app, clusters, provider)
    for p in problems:
        print(f"  found {p['id']}: {p['title']}")
    assign(app, problems, items, provider)

    more = discover_more(app, problems, items, provider)
    for p in more:
        print(f"  found among complaints without a problem {p['id']}: {p['title']}")
    if more:
        assign(app, more, items, provider)
        problems += more

    problems, merged = refine(app, problems, items, provider)
    count(problems, items, clusters, result["trend"])

    kept = sorted((p for p in problems if p["complaints"] >= MIN_PROBLEM_COMPLAINTS), key=lambda p: -p["complaints"])
    kept_ids = {p["id"] for p in kept}
    for item in items:
        if item["problems"] is not None:
            item["problems"] = [pid for pid in item["problems"] if pid in kept_ids]
    for p in kept:
        p["related"] = [r for r in p["related"] if r["id"] in kept_ids]

    current = [i for i in items if i["current"]]
    result.update({
        "cluster_other_share": result.pop("other_share"),
        "other_share": share(sum(not i["problems"] for i in current), len(current)),
        "several_share": share(sum(len(i["problems"]) > 1 for i in current), len(current)),
        "cluster_purity": cluster_purity(items),
        "problems": kept,
        "found_later": [p["title"] for p in more],
        "merged": merged,
        "dropped": [p["title"] for p in problems if p["id"] not in kept_ids],
        "clusters": all_clusters,
        "items": items,
        "llm_provider": provider,
        "llm_model": get_model(provider),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", required=True)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    args = parser.parse_args()

    start = time.perf_counter()
    result = describe_app(args.app, args.days, args.provider)
    titles = {p["id"]: p["title"] for p in result["problems"]}
    print(f"\n{result['app']}, last {result['period_days']} days, {result['complaints']} complaints, "
          f"{len(result['problems'])} problems ({time.perf_counter() - start:.0f} s)")
    print(f"Without any problem: {result['other_share']:.0%}, with several problems: {result['several_share']:.0%}, "
          f"cluster purity: {result['cluster_purity']:.0%}")
    if result["dropped"]:
        print(f"Dropped (fewer than {MIN_PROBLEM_COMPLAINTS} complaints): {'; '.join(result['dropped'])}")
    print()
    for p in result["problems"]:
        direction = {"growing": "↑", "declining": "↓", "stable": "="}.get(p["direction"], " ")
        print(f"{p['complaints']:>4} ({p['share']:.0%}) {direction} {p['title']}")
        print(f"      {p['description']}")
        for quote in p["quotes"]:
            print(f"      > {quote[:160]}")
        for r in p["related"]:
            print(f"      together with «{titles[r['id']]}»: {r['complaints']}")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.app}_described.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()