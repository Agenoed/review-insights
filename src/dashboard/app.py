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
from src.sentiment.baseline import LABELS
from src.sentiment.experiments import METRICS_DIR, MODEL_PATH
from src.sentiment.threshold import apply_threshold

NAMES = {"negative": "негатив", "neutral": "нейтрально", "positive": "позитив"}
COLORS = ["#d9534f", "#adb5bd", "#5cb85c"]
LANGS = ["uk", "ru", "en", "pl", "es"]
TABLE_COLUMNS = ["at", "app", "lang", "score", "confidence", "text"]

st.set_page_config(page_title="Review Insights", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(texts):
    saved = load_model()
    model = saved["model"]
    proba = model.predict_proba(texts)
    return apply_threshold(proba, list(model.classes_), saved["threshold"]), proba.max(axis=1)


@st.cache_data
def load_reviews():
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["at"], keep_default_na=False)
    df["pred"], df["confidence"] = predict(df["text"])
    df["confidence"] = df["confidence"].round(2)
    return df


df = load_reviews()

st.title("Review Insights")
st.caption("Відгуки з Google Play. Тональність визначає TF-IDF + логістична регресія, навчена на LLM-мітках і синтетичних нейтральних відгуках")

with st.sidebar:
    st.header("Фільтри")
    all_apps = sorted(df["app"].unique())
    apps = st.multiselect("Застосунки", all_apps, default=all_apps)
    langs = st.multiselect("Мова", LANGS, default=LANGS)
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Відгуків", f"{len(view):,}".replace(",", " "))
col2.metric("Негативних", f"{(view['pred'] == 'negative').mean():.0%}")
col3.metric("Позитивних", f"{(view['pred'] == 'positive').mean():.0%}")
col4.metric("Нейтральних", f"{(view['pred'] == 'neutral').mean():.0%}")

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
negative = view[view["pred"] == "negative"].sort_values("at", ascending=False).head(20)
st.dataframe(negative[TABLE_COLUMNS], hide_index=True)

st.subheader("Питання і побажання")
st.caption("Нейтральні відгуки: питання, побажання, змішані оцінки. Тут зручно шукати ідеї для продукту")
neutral = view[view["pred"] == "neutral"].sort_values("at", ascending=False).head(20)
st.dataframe(neutral[TABLE_COLUMNS], hide_index=True)

st.subheader("Перевірити відгук")
text = st.text_area("Текст відгуку")
if st.button("Визначити") and text.strip():
    pred, confidence = predict([text])
    st.write(
        f"Тональність: **{NAMES[pred[0]]}** (впевненість {confidence[0]:.2f}), "
        f"мова: **{detect_lang(text)}**"
    )

with st.expander("Якість моделі"):
    results = json.loads((METRICS_DIR / "experiments.json").read_text(encoding="utf-8"))["labels"]
    best = results["LLM labels + synthetic + threshold"]
    st.write(
        f"На тестовій половині ручної розмітки macro-F1 = {best['macro_f1']}, "
        f"F1 нейтральних = {best['f1_per_class']['neutral']}. "
        f"Для порівняння, SVM на зірках: {results['reference: SVM on stars']['macro_f1']}"
    )
    st.json(results)