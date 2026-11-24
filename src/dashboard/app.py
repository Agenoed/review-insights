"""Streamlit dashboard: sentiment overview, complaint problems, report for the product
team and adding a new app by its Google Play link.

Usage:
    python -m streamlit run src/dashboard/app.py
"""

import json
from datetime import timedelta

import pandas as pd
import streamlit as st

from src.llm.report import COUNTRIES, complaints_word, escape, markets_text, percent, period_text, trend_text
from src.pipeline.add_app import COUNTRY_LANGS, LANGS, add_app, read_registry, remove_app
from src.preprocessing.language import detect_lang
from src.sentiment.baseline import LABELS
from src.sentiment.predict import predict
from src.topics.app_topics import OUT_DIR, ROOT, load_reviews

METRICS_DIR = ROOT / "reports" / "metrics"
NAMES = {"negative": "негатив", "neutral": "нейтрально", "positive": "позитив"}
COLORS = ["#d9534f", "#adb5bd", "#5cb85c"]
MODEL_NAMES = {"xlmr": "XLM-RoBERTa, донавчена на наших мітках", "tfidf": "TF-IDF + логістична регресія"}
DIRECTIONS = {"growing": "↑ росте", "declining": "↓ меншає", "stable": "без змін"}
TABLE_COLUMNS = ["at", "app", "lang", "score", "confidence", "text"]
MAX_LINES = 6

st.set_page_config(page_title="Review Insights", layout="wide")


@st.cache_data
def load_data():
    return load_reviews()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def described_apps():
    return sorted(p.name.removesuffix("_described.json") for p in OUT_DIR.glob("*_described.json"))


def app_markets(df, name):
    rows = df[df["app"] == name]
    if not {"country", "request_lang"} <= set(rows.columns):
        return ""
    found = {c: [lang for lang in g["request_lang"].value_counts().index if lang] for c, g in rows.groupby("country") if c}
    return f"Відгуки з Google Play: {markets_text(found)}." if found else ""


def show_overview(df, label):
    c1, c2, c3 = st.columns([2, 1, 1])
    all_apps = sorted(df["app"].unique())
    apps = c1.multiselect("Застосунки", all_apps, default=all_apps, format_func=label)
    langs = c2.multiselect("Мова", LANGS, default=LANGS)
    first_day, last_day = df["at"].min().date(), df["at"].max().date()
    period = c3.date_input("Період", (max(first_day, last_day - timedelta(days=365)), last_day),
                           min_value=first_day, max_value=last_day)
    start = period[0]
    end = period[1] if len(period) == 2 else last_day

    view = df[df["app"].isin(apps) & df["lang"].isin(langs) & df["at"].dt.date.between(start, end)]
    if view.empty:
        st.warning("Немає відгуків під ці фільтри")
        return

    cols = st.columns(4)
    cols[0].metric("Відгуків", f"{len(view):,}".replace(",", " "))
    for col, (key, title) in zip(cols[1:], [("negative", "Негативних"), ("positive", "Позитивних"), ("neutral", "Нейтральних")]):
        col.metric(title, percent((view["pred"] == key).mean()))

    st.subheader("Тональність по застосунках")
    by_app = pd.crosstab(view["app"].map(label), view["pred"], normalize="index")
    st.bar_chart(by_app.reindex(columns=LABELS, fill_value=0).rename(columns=NAMES), color=COLORS)

    st.subheader("Частка негативу по тижнях")
    marked = view.assign(neg=view["pred"] == "negative", name=view["app"].map(label))
    if marked["name"].nunique() <= MAX_LINES:
        weekly = marked.groupby([pd.Grouper(key="at", freq="W"), "name"])["neg"].agg(["mean", "count"])
        st.line_chart(weekly.loc[weekly["count"] >= 10, "mean"].unstack("name"))
        st.caption("Тижні, де в застосунку менше 10 відгуків, не показані")
    else:
        weekly = marked.set_index("at").resample("W")["neg"].agg(["mean", "count"])
        st.line_chart(weekly.loc[weekly["count"] >= 10, "mean"].rename("усі вибрані"))
        st.caption(f"Усі вибрані застосунки разом, тижні з менше ніж 10 відгуками не показані. Застосунки зібрані за різні "
                   f"періоди, тому загальна лінія може стрибати через склад даних. Щоб порівняти застосунки, залиш у фільтрі "
                   f"до {MAX_LINES}")

    st.subheader("Останні негативні відгуки")
    st.dataframe(view[view["pred"] == "negative"].sort_values("at", ascending=False).head(20)[TABLE_COLUMNS], hide_index=True)

    st.subheader("Питання і побажання")
    st.caption("Нейтральні відгуки: питання, побажання, змішані оцінки. Тут зручно шукати ідеї для продукту")
    st.dataframe(view[view["pred"] == "neutral"].sort_values("at", ascending=False).head(20)[TABLE_COLUMNS], hide_index=True)

    st.subheader("Перевірити відгук")
    text = st.text_area("Текст відгуку")
    if st.button("Визначити") and text.strip():
        pred, confidence, model = predict([text])
        st.write(f"Тональність: **{NAMES[pred[0]]}** (впевненість {confidence[0]:.2f}, модель {model}), "
                 f"мова: **{detect_lang(text)}**")

    with st.expander("Модель тональності"):
        counts = df["model"].value_counts()
        st.write("; ".join(f"{MODEL_NAMES.get(m, m)}: {n} відгуків" for m, n in counts.items()))
        metrics = read_json(METRICS_DIR / ("xlmr_finetuned.json" if counts.index[0] == "xlmr" else "experiments.json"))
        if metrics:
            st.json(metrics)


def show_topics(df, name, label):
    result = read_json(OUT_DIR / f"{name}_described.json")
    report = read_json(OUT_DIR / f"{name}_report.json") or {}
    trend = result.get("trend") or {}
    st.subheader(label(name))
    st.caption(f"{app_markets(df, name)} Період: {period_text(result)}. "
               f"Відгуків за період: {result['reviews']}, скарг: {result['complaints']}")

    cols = st.columns(4)
    previous = trend.get("previous_negative_share")
    delta = f"{(result['negative_share'] - previous) * 100:+.0f} п.п." if previous is not None else None
    cols[0].metric("Негативних відгуків", percent(result["negative_share"]), delta=delta, delta_color="inverse")
    cols[1].metric("Проблем", len(result["problems"]))
    cols[2].metric("Скарг без проблеми", percent(result["other_share"]))
    cols[3].metric("Скарг з кількома проблемами", percent(result["several_share"]))

    problems = result["problems"]
    table = pd.DataFrame([{"Проблема": p["title"], "Скарг": p["complaints"], "Частка": round(p["share"] * 100),
                           "Тренд": DIRECTIONS.get(p["direction"], "немає даних")} for p in problems])
    st.dataframe(table, hide_index=True, column_config={
        "Частка": st.column_config.ProgressColumn("Частка скарг", format="%d%%", min_value=0, max_value=100)})
    st.caption("Одна скарга може описувати кілька проблем, тому частки в сумі дають більше 100%")

    titles = {p["id"]: p["title"] for p in problems}
    items = [i for i in result["items"] if i["current"]]
    for p in problems:
        with st.expander(f"{p['title']}: {p['complaints']} {complaints_word(p['complaints'])} ({percent(p['share'])})"):
            st.write(p["description"])
            if p.get("trend"):
                st.caption(trend_text(p))
            info = report.get("problems", {}).get(str(p["id"]), {})
            for quote in info.get("quotes") or p["quotes"]:
                st.markdown(f"> {escape(quote)}")
                if quote in report.get("translations", {}):
                    st.markdown(f"> *Переклад:* {escape(report['translations'][quote])}")
            for r in p["related"]:
                if r["id"] in titles:
                    st.caption(f"Разом із «{titles[r['id']]}»: {r['complaints']} {complaints_word(r['complaints'])}")
            mine = pd.DataFrame([{"Дата": i["at"], "Скарга": i["text"]} for i in items if p["id"] in (i["problems"] or [])])
            if len(mine):
                st.dataframe(mine.sort_values("Дата", ascending=False), hide_index=True)

    with st.expander("Кластери скарг"):
        st.caption("Групи схожих скарг, які знайшов HDBSCAN. Мовна модель бачить їхні типові скарги і за ними складає список проблем")
        st.dataframe(pd.DataFrame([{"Кластер": c["topic"], "Скарг": c["complaints"], "Ключові слова": ", ".join(c["words"][:8])}
                                   for c in result["clusters"]]), hide_index=True)


def show_report(df, name, label):
    path = OUT_DIR / f"{name}_report.md"
    if not path.exists():
        st.info("Звіту для цього застосунку ще немає")
        return
    markdown = path.read_text(encoding="utf-8")
    st.header(f"Скарги користувачів: {label(name)}")
    if "Відгуки з Google Play" not in markdown:
        st.write(app_markets(df, name))
    st.markdown(markdown.split("\n", 1)[1] if markdown.startswith("# ") else markdown)
    st.download_button("Завантажити звіт (.md)", markdown, file_name=f"{name}_report.md", mime="text/markdown")


def run_add(link, name, country, langs, per_lang, use_llm):
    lines, error = [], None
    with st.status("Додаю застосунок", expanded=True) as status:
        def log(message):
            lines.append(message)
            status.write(message)

        try:
            result = add_app(link, name, country, langs, per_lang, use_llm=use_llm, log=log)
        except ValueError as e:
            error = str(e)
            status.error(error)
            status.update(label="Не вийшло", state="error", expanded=True)
    if error:
        return
    load_data.clear()
    st.session_state["added"] = {"result": result, "log": lines}
    st.session_state["pending_app"] = result["name"]
    st.rerun()


def show_add(registry, label):
    removed = st.session_state.pop("removed", None)
    if removed:
        st.success(f"{removed} видалено")
    added = st.session_state.pop("added", None)
    if added:
        result = added["result"]
        if result.get("llm_error"):
            st.warning(f"{result['title']}: відгуки й тональність додано, але проблеми чи звіт не вийшли: {result['llm_error']}")
        else:
            st.success(f"{result['title']} додано: {result['reviews']} відгуків, проблем {result.get('problems', 0)}. "
                       f"Застосунок уже вибрано в бічній панелі, дивись вкладки «Теми скарг» і «Звіт»")
        with st.expander("Що робила система"):
            st.write("\n\n".join(added["log"]))

    st.write("Встав посилання на застосунок у Google Play. Система збере відгуки, визначить тональність, "
             "знайде проблеми у скаргах і напише звіт. Це займає кілька хвилин.")
    country = st.selectbox("Країна магазину Google Play", list(COUNTRY_LANGS), format_func=lambda c: f"{COUNTRIES[c]} ({c})",
                           help="Регіон магазину. Мови нижче підставляються під нього. Google Play віддає відгуки "
                                "насамперед за мовою, тому англомовні відгуки можуть бути від користувачів з різних країн")
    with st.form("add"):
        link = st.text_input("Посилання або id застосунку", placeholder="https://play.google.com/store/apps/details?id=...")
        c1, c2 = st.columns(2)
        name = c1.text_input("Назва в системі (необов'язково)",
                             help="Якщо порожньо, назва складеться з назви в Google Play. Великі літери й кирилиця "
                                  "перетворяться на малі латинські, пробіли на _")
        per_lang = c2.number_input("Відгуків на кожну мову", min_value=100, max_value=3000, value=1000, step=100)
        langs = st.multiselect("Мови відгуків", LANGS, default=COUNTRY_LANGS[country],
                               help="Якими мовами написані відгуки, які збирати. Модель тональності знає лише ці п'ять мов")
        use_llm = st.checkbox("Знайти проблеми і написати звіт (потрібен ключ мовної моделі)", value=True)
        submitted = st.form_submit_button("Додати")
    if submitted and link.strip():
        run_add(link, name.strip() or None, country, langs, int(per_lang), use_llm)

    if not registry:
        return
    st.subheader("Додані застосунки")
    st.dataframe(pd.DataFrame([{
        "Застосунок": label(name), "Країна": COUNTRIES.get(info["country"], info["country"]),
        "Мови": ", ".join(info["languages"]), "Відгуків": info["reviews"], "Негативних": percent(info["negative_share"]),
        "Проблем": info.get("problems"), "Відгуки оновлено": info["updated"].replace("T", " "),
        "Звіт оновлено": (info.get("report_updated") or "").replace("T", " "), "Помилка": info.get("llm_error") or "",
    } for name, info in registry.items()]), hide_index=True)
    with st.form("update"):
        chosen = st.selectbox("Дозібрати нові відгуки й оновити звіт", list(registry), format_func=label)
        update = st.form_submit_button("Оновити")
    if update:
        info = registry[chosen]
        run_add(info["link"], chosen, info["country"], info["languages"], info["per_lang"], True)

    with st.form("remove"):
        doomed = st.selectbox("Видалити застосунок", list(registry), format_func=label)
        sure = st.checkbox("Так, видалити його відгуки, теми і звіт назавжди")
        remove = st.form_submit_button("Видалити")
    if remove and not sure:
        st.warning("Постав галочку, щоб підтвердити видалення")
    elif remove:
        info = remove_app(doomed)
        load_data.clear()
        st.session_state["removed"] = info["title"]
        st.rerun()


df = load_data()
registry = read_registry()


def label(name):
    return f"{registry[name]['title']} ({name})" if name in registry else name


st.title("Review Insights")
model = df["model"].mode()[0]
st.caption(f"Відгуки з Google Play. Тональність визначає {MODEL_NAMES.get(model, model)}, "
           "проблеми у скаргах шукає мовна модель за кластерами схожих скарг")

apps_with_topics = described_apps()
pending = st.session_state.pop("pending_app", None)
if pending in apps_with_topics:
    st.session_state["app"] = pending
if st.session_state.get("app") not in apps_with_topics:
    st.session_state.pop("app", None)
chosen = None
with st.sidebar:
    st.header("Застосунок")
    if apps_with_topics:
        chosen = st.selectbox("Для вкладок «Теми скарг» і «Звіт»", apps_with_topics, format_func=label, key="app")

overview, topics, report, add = st.tabs(["Огляд", "Теми скарг", "Звіт", "Додати застосунок"])
with overview:
    show_overview(df, label)
with topics:
    if chosen:
        show_topics(df, chosen, label)
    else:
        st.info("Ще немає жодного застосунку з розібраними скаргами")
with report:
    if chosen:
        show_report(df, chosen, label)
    else:
        st.info("Ще немає жодного звіту")
with add:
    show_add(registry, label)