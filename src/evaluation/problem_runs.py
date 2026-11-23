"""Repeated runs of the problem search, for stability and for H3.

`run` repeats describe_app for one app and saves every run to data/apps/runs.
With --groups random the clusters are replaced by random groups of the same sizes,
so the LLM sees the same number of complaints, but they are not grouped by meaning.
The prompts stay the same. `compare` matches problems between runs by the complaints
they got and saves the metrics. The main {app}_described.json counts as the first
clusters run and is never overwritten.

Usage:
    python -m src.evaluation.problem_runs run --app glovo --groups clusters --runs 2
    python -m src.evaluation.problem_runs run --app glovo --groups random --runs 3
    python -m src.evaluation.problem_runs compare
"""

import argparse
import json
import random
import time
from itertools import permutations
from pathlib import Path

import numpy as np

from src.llm.topics import describe_app
from src.topics.app_topics import OUT_DIR
from src.topics.modeling import shorten

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = OUT_DIR / "runs"
METRICS_PATH = ROOT / "reports" / "metrics" / "problem_runs.json"
MATCH_LIMIT = 0.5
BIG_SHARE = 0.10
EXAMPLES = 25
RANDOM_TOPIC = 1000


def random_groups(seed):
    def make(clusters, items):
        current = [i for i in items if i["current"]]
        random.Random(seed).shuffle(current)
        groups, start = [], 0
        for k, cluster in enumerate(clusters):
            part = current[start:start + cluster["complaints"]]
            start += cluster["complaints"]
            groups.append({"topic": RANDOM_TOPIC + k, "complaints": len(part),
                           "examples": [shorten(i["text"]) for i in part[:EXAMPLES]]})
        return groups
    return make


def run(app, groups, runs, provider=None):
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    done = len(list(RUNS_DIR.glob(f"{app}_{groups}_*.json")))
    for k in range(done + 1, done + runs + 1):
        start = time.perf_counter()
        make_groups = random_groups(k) if groups == "random" else None
        result = describe_app(app, provider=provider, make_groups=make_groups)
        result["groups"] = groups
        path = RUNS_DIR / f"{app}_{groups}_{k}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{app} {groups} run {k}: {len(result['problems'])} problems, without a problem {result['other_share']:.0%}, "
              f"{time.perf_counter() - start:.0f} s, saved to {path}")


def load_runs(app):
    runs = {"clusters": [], "random": []}
    first = OUT_DIR / f"{app}_described.json"
    if first.exists():
        runs["clusters"].append(json.loads(first.read_text(encoding="utf-8")))
    for groups in runs:
        for path in sorted(RUNS_DIR.glob(f"{app}_{groups}_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            runs[groups].append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def complaint_sets(result):
    current = [i for i in result["items"] if i["current"]]
    return [({i["id"] for i in current if p["id"] in (i["problems"] or [])}, p["share"]) for p in result["problems"]]


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def matched(a, b):
    found = []
    for ids, problem_share in a:
        best = max((jaccard(ids, other) for other, _ in b), default=0.0)
        found.append((best >= MATCH_LIMIT, len(ids), problem_share))
    return found


def agreement(pairs):
    found = [m for a, b in pairs for m in matched(a, b)]
    big = [ok for ok, _, s in found if s >= BIG_SHARE]
    return {
        "problems_matched": round(float(np.mean([ok for ok, _, _ in found])), 3) if found else None,
        "complaints_matched": round(sum(n for ok, n, _ in found if ok) / max(1, sum(n for _, n, _ in found)), 3),
        "big_problems_matched": round(float(np.mean(big)), 3) if big else None,
        "pairs": len(pairs),
    }


def run_stats(result):
    return {
        "problems": len(result["problems"]),
        "other_share": result["other_share"],
        "several_share": result["several_share"],
        "merged": len(result["merged"]),
        "found_later": len(result["found_later"]),
        "dropped": len(result["dropped"]),
    }


def mean_stats(stats):
    return {k: round(float(np.mean([s[k] for s in stats])), 3) for k in stats[0]}


def compare():
    results = {}
    for app in sorted({p.stem.rsplit("_", 2)[0] for p in RUNS_DIR.glob("*.json")}):
        runs = load_runs(app)
        ids = [frozenset(i["id"] for i in r["items"] if i["current"]) for rs in runs.values() for r in rs]
        if len(set(ids)) > 1:
            print(f"{app}: the runs have different complaints, maybe the data changed between them")
        sets = {groups: [complaint_sets(r) for r in rs] for groups, rs in runs.items()}
        app_result = {}
        for groups, rs in runs.items():
            if not rs:
                continue
            app_result[groups] = {
                "runs": len(rs),
                "mean": mean_stats([run_stats(r) for r in rs]),
                "each": [run_stats(r) for r in rs],
                "stability": agreement(list(permutations(sets[groups], 2))) if len(rs) > 1 else None,
                "titles": [[p["title"] for p in r["problems"]] for r in rs],
            }
        if sets["clusters"] and sets["random"]:
            app_result["clusters_found_by_random"] = agreement([(c, r) for c in sets["clusters"] for r in sets["random"]])
            app_result["random_found_by_clusters"] = agreement([(r, c) for c in sets["clusters"] for r in sets["random"]])
        results[app] = app_result
        for groups in ("clusters", "random"):
            if groups in app_result:
                print(app, groups, app_result[groups]["runs"], "runs", app_result[groups]["mean"],
                      "stability", app_result[groups]["stability"])
        if "clusters_found_by_random" in app_result:
            print(app, "clusters found by random", app_result["clusters_found_by_random"])
            print(app, "random found by clusters", app_result["random_found_by_clusters"])

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "compare"])
    parser.add_argument("--app")
    parser.add_argument("--groups", choices=["clusters", "random"], default="clusters")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    args = parser.parse_args()
    if args.command == "run":
        if not args.app:
            parser.error("run needs --app")
        run(args.app, args.groups, args.runs, args.provider)
    else:
        compare()


if __name__ == "__main__":
    main()