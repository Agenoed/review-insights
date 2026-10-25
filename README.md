# review-insights

Магістерська робота (Data Science & Data Analytics) – система аналізу тональності багатомовних відгуків клієнтів.

Автор: Олександр Деонега, керівник: Олексій Кудін

Ідея: збирати відгуки про застосунки з Google Play (українською, російською, англійською), визначати тональність, групувати скарги за темами і через LLM робити з цього короткий звіт для продуктової команди.

## Запуск

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python -m src.data.collect_google_play --per-lang 500

CSV зберігається в data/raw (папка в .gitignore).

## Що є зараз

- `src/data/collect_google_play.py` – збір відгуків, список застосунків у `configs/apps.json`
- `notebooks/01_eda.ipynb` – перший огляд даних
- `src/preprocessing/prepare.py` – обробка: очищення, визначення мови, розбиття на train/val/test
- `src/evaluation/checks.py` – перевірка зірок і детекторів мови на ручній розмітці
- `src/sentiment/baseline.py` – бейзлайн TF-IDF + SVM, метрики в `reports/metrics`
- `src/dashboard/app.py` – дашборд, запуск: `python -m streamlit run src/dashboard/app.py`

Далі: трансформери (XLM-RoBERTa) і тематичне моделювання скарг.