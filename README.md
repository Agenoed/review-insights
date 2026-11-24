# review-insights

Магістерська робота (Data Science & Data Analytics) – система аналізу тональності багатомовних відгуків клієнтів.

Автор: Олександр Деонега, керівник: Олексій Кудін

Система збирає відгуки про застосунки з Google Play п'ятьма мовами (українська, російська, англійська, польська, іспанська), визначає тональність, знаходить, на що саме скаржаться користувачі, і пише звіт для продуктової команди. Новий застосунок можна додати за посиланням з Google Play прямо з дашборду або через API.

## Що вміє

- збір відгуків з Google Play для різних країн і мов
- очищення тексту і визначення мови без перекладу
- тональність (негатив / нейтрально / позитив) з упевненістю моделі
- пошук проблем у скаргах на конкретний застосунок: назва, опис, скільки скарг, чи росте проблема, цитати
- звіт: деталі кожної проблеми, які проблеми йдуть разом, що варто зробити, переклад цитат
- дашборд на Streamlit і REST API на FastAPI
- запуск усього через Docker

## Основні результати

- Тональність: донавчена XLM-RoBERTa дає macro-F1 0,837 на тестовій половині ручної розмітки (301 відгук). TF-IDF на тих самих мітках – 0,802, бейзлайн TF-IDF + SVM на зірках – 0,661. Швидкість: 0,7 с на 1000 відгуків на відеокарті, близько хвилини на процесорі.
- Мітки від LLM збігаються з моєю ручною розміткою краще, ніж зірки: kappa 0,92 у Gemma 4 26B через OpenRouter і 0,87 у локальної Gemma 3 12B через Ollama, у зірок 0,69.
- Пошук проблем перевірено вручну на 120 скаргах Дії, Glovo і monobank: F1 0,65 при сліпій розмітці і 0,91 після перегляду розбіжностей. Деталі у звітах підтверджуються самими скаргами в 91% випадків.

Усі метрики лежать у `reports/metrics`.

## Запуск через Docker

Потрібен Docker Desktop. Папок `data/` і `models/` у репозиторії немає, їх треба мати локально (вони створюються кроками пайплайну нижче). Контейнери підключають ці папки з диска.

1. Створити `.env` у корені проєкту:

   ```
   LLM_PROVIDER=openrouter
   OPENROUTER_API_KEY=твій_ключ
   OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
   ```

2. Запустити:

   ```
   docker compose up --build
   ```

3. Дашборд: http://localhost:8501, API з документацією: http://localhost:8000/docs

У контейнері немає відеокарти, тому XLM-R працює на процесорі, приблизно хвилина на 1000 відгуків. Модель для ембедингів завантажується при першому пошуку тем і зберігається в томі `hf-cache`. Файл `.env` в образ не потрапляє, контейнери читають його при запуску. Зупинити: `docker compose down`.

## Локальний запуск

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    pip install torch --index-url https://download.pytorch.org/whl/cu128

PyTorch ставиться окремо, бо збірка залежить від відеокарти (у мене RTX 5070, CUDA 12.8). Без відеокарти можна поставити звичайний `pip install torch`.

Файл `.env` (в git не потрапляє):

    LLM_PROVIDER=openrouter
    OPENROUTER_API_KEY=...
    OPENROUTER_MODEL=google/gemma-4-26b-a4b-it
    OLLAMA_MODEL=gemma3:12b
    SENTIMENT_MODEL=auto

`LLM_PROVIDER=ollama` – локальна модель замість хмарної. `SENTIMENT_MODEL`: `auto` бере XLM-R, якщо є відеокарта, інакше TF-IDF; можна задати `xlmr` або `tfidf` явно.

Дашборд і API:

    python -m streamlit run src/dashboard/app.py
    uvicorn src.api.main:app --reload

Для них потрібні вже підготовлені дані й моделі, тобто кроки нижче.

## Весь пайплайн по порядку

Дані:

    python -m src.data.collect_google_play --per-lang 2000
    python -m src.data.collect_google_play --config configs/apps_extra.json --per-lang 1000
    python -m src.preprocessing.prepare

`prepare` створює `data/labeling/to_label.xlsx` для ручної розмітки, правила в `docs/labeling_guide.md`. Після розмітки:

    python -m src.evaluation.checks

LLM-мітки і синтетичні нейтральні відгуки:

    python -m src.llm.label --source gold
    python -m src.llm.label --source train --n 5000
    python -m src.llm.synthetic --per-combo 6

Моделі тональності:

    python -m src.sentiment.baseline
    python -m src.sentiment.threshold
    python -m src.sentiment.experiments
    python -m src.sentiment.zero_shot
    python -m src.sentiment.finetune_xlmr
    python -m src.sentiment.finetune_xlmr --base cardiffnlp/twitter-xlm-roberta-base-sentiment --name xlmt_finetuned
    python -m src.pipeline.score
    python -m src.evaluation.benchmark

`score` один раз оцінює всі відгуки і зберігає результат, дашборд, теми й API потім читають готовий файл.

Теми скарг і проблеми:

    python -m src.topics.compare
    python -m src.evaluation.topic_languages
    python -m src.llm.topics --app glovo
    python -m src.llm.report --app glovo

`compare` порівнює BERTopic, k-means і LDA на всіх скаргах, `topic_languages` рахує, чи змішуються мови всередині тем. `llm.topics` шукає проблеми для одного застосунку, `llm.report` пише звіт.

Перевірка знайдених проблем:

    python -m src.evaluation.problem_check make --app glovo
    python -m src.evaluation.problem_check review --app glovo
    python -m src.evaluation.problem_check score
    python -m src.evaluation.problem_runs run --app glovo --groups clusters --runs 2
    python -m src.evaluation.problem_runs run --app glovo --groups random --runs 3
    python -m src.evaluation.problem_runs compare

`make` готує Excel для сліпої ручної розмітки, `review` – файл з розбіжностями, `score` рахує метрики з довірчими інтервалами. `problem_runs` перевіряє стабільність повторних запусків і порівнює кластери з випадковими групами.

## Додати застосунок

У дашборді вкладка «Додати застосунок», або з командного рядка:

    python -m src.pipeline.add_app "https://play.google.com/store/apps/details?id=com.ubercab"
    python -m src.pipeline.add_app com.ubercab --country pl --langs pl en
    python -m src.pipeline.add_app --remove uber

Повторний запуск для того самого застосунку дозбирає нові відгуки й оновлює звіт. Додані застосунки зберігаються окремо в `data/added`, основний набір і поділ на train/val/test не змінюються.

## API

- `GET /health` – перевірка, що сервер працює
- `POST /sentiment` – тональність і мова для довільних текстів
- `GET /apps` – застосунки, для яких є дані
- `GET /apps/{name}/problems` – проблеми в скаргах
- `GET /apps/{name}/problems/{id}` – одна проблема з усіма її скаргами
- `GET /apps/{name}/report` – звіт
- `POST /apps` – додати застосунок (працює у фоні, повертає номер завдання)
- `POST /apps/{name}/update` – дозібрати нові відгуки
- `DELETE /apps/{name}` – видалити доданий застосунок
- `GET /jobs/{id}` – стан завдання

Повний опис з прикладами на http://localhost:8000/docs

## Тести

    pytest

67 тестів, проходять за кілька секунд. У мережу не ходять: Google Play і LLM у тестах підмінені.

## Структура

- `src/data/` – збір відгуків з Google Play
- `src/preprocessing/` – очищення, визначення мови, розбиття на train/val/test
- `src/llm/` – клієнт для OpenRouter і Ollama, LLM-розмітка, синтетика, пошук проблем, звіт
- `src/sentiment/` – моделі тональності: бейзлайн, поріг, порівняння міток, XLM-T, донавчання XLM-RoBERTa
- `src/topics/` – ембединги, кластеризація, порівняння методів, теми скарг одного застосунку
- `src/pipeline/` – оцінка всіх відгуків і додавання застосунку
- `src/evaluation/` – перевірки на ручній розмітці, швидкодія, перевірка проблем, повторні запуски
- `src/dashboard/app.py` – дашборд
- `src/api/main.py` – REST API
- `configs/` – застосунки, країни й мови для збору
- `docs/` – правила розмітки, опис архітектури
- `reports/` – метрики, графіки, таблиці тем
- `tests/` – автотести

Дані (`data/`) і моделі (`models/`) в репозиторій не потрапляють. Ключ до OpenRouter зберігається лише в `.env`.