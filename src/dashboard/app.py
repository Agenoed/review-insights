"""Streamlit dashboard for Google Play reviews.

Usage:
    python -m streamlit run src/dashboard/app.py
"""

import json
from datetime import timedelta

import joblib
import pandas as pd
import streamlit as st

from src.preprocessing.language import detect_lang
from src.preprocessing.prepare import PROCESSED_PATH
from src.sentiment.baseline import LABELS, METRICS_PATH, MODEL_PATH

NAMES = {"negative": "негатив", "neutral": "нейтрально", "positive": "позитив"}
COLORS = ["#d9534f", "#adb5bd", "#5cb85c"]

st.set_page_config(page_title="Review Insights", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reviews():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["at"], keep_default_na=False)
    df["pred"] = load_model().predict(df["text"])
    return df


df = load_reviews()

st.title("Review Insights")
st.caption("Відгуки з Google Play, тональність визначає бейзлайн TF-IDF + SVM")

with st.sidebar:
    st.header("Фільтри")
    all_apps = sorted(df["app"].unique())
    apps = st.multiselect("Застосунки", all_apps, default=all_apps)
    langs = st.multiselect("Мова", ["uk", "ru", "en", "mixed"], default=["uk", "ru", "en", "mixed"])
    first_day = df["at"].min().date()
    last_day = df["at"].max().date()
    period = st.date_input(
        "Період",
        (max(first_day, last_day - timedelta(days=365)), last_day),
        min_value=first_day,
        max_value=last_day,
    )

start = period[0]
end = period[1] if len(period) == 2 else last_day

view = df[
    df["app"].isin(apps)
    & df["lang"].isin(langs)
    & df["at"].dt.date.between(start, end)
]
if view.empty:
    st.warning("Немає відгуків під ці фільтри")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Відгуків", f"{len(view):,}".replace(",", " "))
col2.metric("Негативних", f"{(view['pred'] == 'negative').mean():.0%}")
col3.metric("Позитивних", f"{(view['pred'] == 'positive').mean():.0%}")

st.subheader("Тональність по застосунках")
by_app = pd.crosstab(view["app"], view["pred"], normalize="index")
by_app = by_app.reindex(columns=LABELS, fill_value=0).rename(columns=NAMES)
st.bar_chart(by_app, color=COLORS)

st.subheader("Частка негативу по тижнях")
weekly = (
    view.assign(neg=view["pred"] == "negative")
    .set_index("at")
    .resample("W")["neg"]
    .agg(["mean", "count"])
)
weekly = weekly[weekly["count"] >= 10]
st.line_chart(weekly["mean"].rename("негатив"))
st.caption("Тижні, де менше 10 відгуків, не показані")

st.subheader("Останні негативні відгуки")
latest = view[view["pred"] == "negative"].sort_values("at", ascending=False).head(20)
st.dataframe(latest[["at", "app", "lang", "score", "text"]], hide_index=True)

st.subheader("Перевірити відгук")
text = st.text_area("Текст відгуку")
if st.button("Визначити") and text.strip():
    sentiment = load_model().predict([text])[0]
    st.write(f"Тональність: **{NAMES[sentiment]}**, мова: **{detect_lang(text)}**")

with st.expander("Якість моделі"):
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    gold = metrics["gold_manual"]
    st.write(f"На ручній розмітці (n={gold['n']}) macro-F1 = {gold['macro_f1']}")
    st.json(metrics)