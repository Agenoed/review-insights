# review-insights

Магістерська робота (Data Science & Data Analytics) – система аналізу тональності багатомовних відгуків клієнтів.

Автор: Олександр Деонега, керівник: Олексій Кудін

Ідея: збирати відгуки про застосунки з Google Play різними мовами, визначати тональність, групувати скарги за темами і через LLM робити з цього короткий звіт для продуктової команди.

## Запуск

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cu128

Для LLM потрібен файл `.env` (в git не потрапляє):

    LLM_PROVIDER=openrouter
    OPENROUTER_API_KEY=...
    OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
    OLLAMA_MODEL=gemma3:12b

Порядок запуску:

    python -m src.data.collect_google_play --per-lang 2000
    python -m src.data.collect_google_play --config configs/apps_extra.json --per-lang 1000
    python -m src.preprocessing.prepare
    python -m src.llm.label --source gold
    python -m src.llm.label --source train --n 5000
    python -m src.llm.synthetic --per-combo 6
    python -m src.sentiment.baseline
    python -m src.sentiment.threshold
    python -m src.sentiment.experiments
    python -m src.sentiment.zero_shot
    python -m streamlit run src/dashboard/app.py

Тести: `pytest`

## Що є зараз

- `src/data/` – збір відгуків з Google Play
- `src/preprocessing/` – очищення, визначення мови, розбиття на train/val/test
- `src/llm/` – клієнт для OpenRouter і Ollama, LLM-розмітка, синтетичні нейтральні відгуки
- `src/sentiment/` – моделі тональності: бейзлайн, поріг, порівняння міток, XLM-T без донавчання
- `src/evaluation/` – перевірки на ручній розмітці, замір швидкодії
- `src/dashboard/app.py` – дашборд
- `docs/` – правила розмітки, опис архітектури
- `reports/` – метрики і графіки
- `tests/` – автотести

Дані (`data/`) і моделі (`models/`) в репозиторій не потрапляють.

Далі: донавчання XLM-RoBERTa, тематичне моделювання скарг, LLM-агент для звітів.