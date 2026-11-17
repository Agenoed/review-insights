"""Manual check of the problems found by the LLM.

`make` builds an Excel file for one app with three sheets: the list of problems,
a random sample of complaints to label by hand (which problems each complaint
has) and the details from the report together with the complaints they were
taken from. The model answers go to a separate csv, so the labeling is blind.
`review` makes a second file with the complaints where the labels and the model
disagree. Both answers are shown in random order without saying whose is whose.
`score` compares the labels with the model, before and after the review.

Usage:
    python -m src.evaluation.problem_check make --app glovo
    python -m src.evaluation.problem_check review --app glovo
    python -m src.evaluation.problem_check score
"""

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment

from src.llm.client import get_provider
from src.llm.report import translate
from src.llm.topics import ASSIGN_CHARS, WORKERS
from src.topics.app_topics import OUT_DIR
from src.topics.modeling import shorten

ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = ROOT / "data" / "labeling"
METRICS_PATH = ROOT / "reports" / "metrics" / "problem_check.json"
SAMPLE = 40
DETAILS_PER_PROBLEM = 2
COMPLAINTS_PER_DETAIL = 3
TRANSLATE_BATCH = 15
SEED = 42
WIDTHS = {"title": 50, "description": 90, "text": 80, "translation": 80, "detail": 50, "correct": 12, "supported": 12,
          "note": 30, "variant_a": 45, "variant_b": 45, "final": 12}
TEXT_COLUMNS = ("correct", "supported", "final")


def translate_all(texts, provider):
    texts = list(dict.fromkeys(texts))
    batches = [texts[i:i + TRANSLATE_BATCH] for i in range(0, len(texts), TRANSLATE_BATCH)]
    with ThreadPoolExecutor(WORKERS) as pool:
        parts = list(pool.map(lambda batch: translate(batch, provider), batches))
    return {k: v for part in parts for k, v in part.items()}


def write_sheets(path, sheets):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for k, column in enumerate(df.columns, 1):
                letter = ws.cell(row=1, column=k).column_letter
                ws.column_dimensions[letter].width = WIDTHS.get(column, 10)
                for cell in ws[letter][1:]:
                    cell.alignment = Alignment(wrap_text=True, vertical="top")
                    if column in TEXT_COLUMNS:
                        cell.number_format = "@"
            ws.freeze_panes = "A2"


def make(app, provider=None, sample=SAMPLE, force=False):
    out = LABEL_DIR / f"problems_{app}.xlsx"
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists, it may have your labels. Use --force to make it again")
    provider = get_provider(provider)
    described = json.loads((OUT_DIR / f"{app}_described.json").read_text(encoding="utf-8"))
    report = json.loads((OUT_DIR / f"{app}_report.json").read_text(encoding="utf-8"))
    problems = described["problems"]
    number = {p["id"]: k for k, p in enumerate(problems, 1)}
    by_id = {i["id"]: i for i in described["items"]}

    current = [i for i in described["items"] if i["current"]]
    picked = random.Random(SEED).sample(current, min(sample, len(current)))
    label = pd.DataFrame([{"n": k, "text": shorten(i["text"], ASSIGN_CHARS), "id": i["id"]}
                          for k, i in enumerate(picked, 1)])
    model = pd.DataFrame([{"id": i["id"], "model": ", ".join(str(number[p]) for p in i["problems"] if p in number)}
                          for i in picked])

    rows = []
    for pid, info in report["problems"].items():
        for detail in info["details"][:DETAILS_PER_PROBLEM]:
            if "ids" not in detail:
                raise ValueError(f"no complaint ids in {app}_report.json, run src.llm.report again")
            for cid in detail["ids"][:COMPLAINTS_PER_DETAIL]:
                rows.append({"problem": number.get(int(pid)), "detail": detail["text"], "text": shorten(by_id[cid]["text"])})
    details = pd.DataFrame(rows)

    translations = translate_all(list(label["text"]) + list(details["text"] if len(details) else []), provider)
    label.insert(2, "translation", label["text"].map(translations).fillna(""))
    label.insert(3, "correct", "")
    label.insert(4, "note", "")
    if len(details):
        details.insert(3, "translation", details["text"].map(translations).fillna(""))
        details.insert(4, "supported", "")

    listing = pd.DataFrame([{"number": k, "title": p["title"], "description": p["description"], "complaints": p["complaints"]}
                            for k, p in enumerate(problems, 1)])
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    write_sheets(out, {"problems": listing, "label": label, "details": details})
    model.to_csv(LABEL_DIR / f"problems_{app}_model.csv", index=False)
    print(f"{app}: {len(label)} complaints and {len(details)} detail checks, saved to {out}")


def numbers(value):
    if pd.isna(value) or not str(value).strip():
        return None
    return {int(x) for x in re.findall(r"\d+", str(value))} - {0}


def read_model(app):
    model = pd.read_csv(LABEL_DIR / f"problems_{app}_model.csv", dtype=str, keep_default_na=False)
    return dict(zip(model["id"], model["model"]))


def variant(found, titles):
    if not found:
        return "0 — жодної проблеми зі списку"
    return "\n".join(f"{k} — {titles.get(k, '?')}" for k in sorted(found))


def review(app, force=False):
    out = LABEL_DIR / f"review_{app}.xlsx"
    if out.exists() and not force:
        raise SystemExit(f"{out} already exists, it may have your answers. Use --force to make it again")
    path = LABEL_DIR / f"problems_{app}.xlsx"
    problems = pd.read_excel(path, sheet_name="problems", dtype=str)
    titles = {int(k): t for k, t in zip(problems["number"], problems["title"])}
    label = pd.read_excel(path, sheet_name="label", dtype=str).fillna("")
    model = read_model(app)
    rng = random.Random(SEED)
    rows = []
    for r in label.itertuples():
        gold, pred = numbers(r.correct), numbers(model.get(r.id)) or set()
        if gold is None or gold == pred:
            continue
        a, b = (gold, pred) if rng.random() < 0.5 else (pred, gold)
        rows.append({"n": r.n, "text": r.text, "translation": r.translation, "variant_a": variant(a, titles),
                     "variant_b": variant(b, titles), "final": "", "note": "", "id": r.id})
    write_sheets(out, {"problems": problems, "review": pd.DataFrame(rows)})
    print(f"{app}: {len(rows)} complaints to review, saved to {out}")


def apply_review(app, label):
    path = LABEL_DIR / f"review_{app}.xlsx"
    if not path.exists():
        return label
    review = pd.read_excel(path, sheet_name="review", dtype=str)
    final = {i: v for i, v in zip(review["id"], review["final"]) if numbers(v) is not None}
    original = dict(zip(label["id"], label["correct"]))
    changed = sum(numbers(v) != numbers(original.get(i)) for i, v in final.items())
    print(f"{app}: reviewed {len(final)} of {len(review)} disagreements, changed {changed} labels")
    label = label.copy()
    label["correct"] = [final.get(i, c) for i, c in zip(label["id"], label["correct"])]
    return label


def collect(app, label, model, n_problems):
    pairs = []
    for n, i, gold in zip(label["n"], label["id"], label["correct"].map(numbers)):
        if gold is None:
            continue
        if any(k > n_problems for k in gold):
            print(f"{app}: complaint {n} has a problem number above {n_problems}, check it")
        pairs.append((gold, numbers(model.get(i)) or set()))
    return pairs


def counts(pairs):
    tp = sum(len(gold & pred) for gold, pred in pairs)
    fp = sum(len(pred - gold) for gold, pred in pairs)
    fn = sum(len(gold - pred) for gold, pred in pairs)
    return tp, fp, fn


def prf(pairs):
    tp, fp, fn = counts(pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return precision, recall, f1


def error_kind(gold, pred):
    if gold == pred:
        return "same"
    if not gold:
        return "model_problem_where_none"
    if not pred:
        return "model_none"
    if gold < pred:
        return "model_extra_problem"
    if pred < gold:
        return "model_missed_problem"
    if not gold & pred:
        return "model_other_problem"
    return "model_partly_other"


def pair_scores(pairs):
    return dict(zip(["precision", "recall", "f1"], prf(pairs)))


def supported_share(values):
    return {"details_supported": sum(values) / len(values)}


def bootstrap(values, stat, n=1000):
    rng = np.random.default_rng(SEED)
    draws = [stat([values[i] for i in rng.integers(0, len(values), len(values))]) for _ in range(n)]
    result = {}
    for name in draws[0]:
        v = [d[name] for d in draws if d[name] is not None]
        if v:
            result[name] = [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]
    return result


def rounded(value):
    return round(value, 3) if value is not None else None


def summary(pairs, supported):
    precision, recall, f1 = prf(pairs)
    ci = {}
    if pairs:
        ci.update(bootstrap(pairs, pair_scores))
    if supported:
        ci.update(bootstrap(supported, supported_share))
    return {
        "complaints": len(pairs),
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1) if pairs else None,
        "exact_match": rounded(sum(gold == pred for gold, pred in pairs) / len(pairs)) if pairs else None,
        "no_problem_agreement": rounded(sum((not gold) == (not pred) for gold, pred in pairs) / len(pairs)) if pairs else None,
        "errors": dict(Counter(error_kind(gold, pred) for gold, pred in pairs).most_common()),
        "details_checked": len(supported),
        "details_supported": rounded(sum(supported) / len(supported)) if supported else None,
        "ci95": ci,
    }


def score():
    results = {"blind": {}, "reviewed": {}}
    pooled = {"blind": [], "reviewed": []}
    all_supported = []
    for path in sorted(LABEL_DIR.glob("problems_*.xlsx")):
        app = path.stem.removeprefix("problems_")
        n_problems = len(pd.read_excel(path, sheet_name="problems"))
        label = pd.read_excel(path, sheet_name="label", dtype=str)
        model = read_model(app)
        details = pd.read_excel(path, sheet_name="details", dtype=str)
        supported = [int(v) for v in details.get("supported", pd.Series(dtype=str)).dropna() if str(v).strip() in ("0", "1")]
        print(f"{app}: labeled {label['correct'].map(numbers).notna().sum()} of {len(label)} complaints "
              f"and {len(supported)} of {len(details)} details")
        all_supported += supported
        for name, table in (("blind", label), ("reviewed", apply_review(app, label))):
            pairs = collect(app, table, model, n_problems)
            if not pairs and not supported:
                continue
            results[name][app] = summary(pairs, supported)
            pooled[name] += pairs

    for name, pairs in pooled.items():
        if pairs or all_supported:
            results[name]["all"] = summary(pairs, all_supported)
        for app, values in results[name].items():
            short = {k: values[k] for k in ("complaints", "precision", "recall", "f1", "exact_match", "details_supported")}
            print(name, app, short)

    if pooled["blind"] or all_supported:
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved to {METRICS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["make", "review", "score"])
    parser.add_argument("--app")
    parser.add_argument("--sample", type=int, default=SAMPLE)
    parser.add_argument("--provider", choices=["openrouter", "ollama"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command in ("make", "review") and not args.app:
        parser.error(f"{args.command} needs --app")
    if args.command == "make":
        make(args.app, args.provider, args.sample, args.force)
    elif args.command == "review":
        review(args.app, args.force)
    else:
        score()


if __name__ == "__main__":
    main()