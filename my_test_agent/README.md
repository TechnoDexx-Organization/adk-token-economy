# my_test_agent — экспериментальный ADK-агент по экономии токенов

Экспериментальный агент на **Google ADK** + **LiteLLM**, цель которого —
снижение расхода токенов при длинных диалогах без потери качества ответа.
Агент формально является «эхо-агентом» (повторяет текст пользователя), что
позволяет чисто измерять эффект от оптимизаций контекста.

## Что делает агент

Ключевые механизмы (см. `agent.py`):

1. **Скользящее окно истории** (`HISTORY_WINDOW_SIZE = 10`) — в LLM-запрос
   уходят только последние 10 событий вместо всей сессии.
   Ссылка на эксперимент: «A sliding window has been added (the hypothesis has been proven)».
2. **Сушка текста** (`sparsify_text`) — удаляются слова-связки и «вода»
   («короче», «в общем», «типа», …), сложные предложения разбиваются на
   короткие. Меньше токенов — тот же смысл.
3. **Эмоциональный анализ** — модель `Aniemore/rubert-tiny2-russian-emotion-detection`
   (HuggingFace transformers) определяет эмоцию пользователя; она сохраняется
   в `state["current_emotion"]`, а в конец последнего сообщения добавляется
   `[Tone hint: ...]` из словаря `TONE_HINTS`.
4. **Статичная инструкция** (`get_static_instruction`) — системный промпт
   не содержит динамики, чтобы провайдер мог кэшировать его и не списывать
   токены повторно.
5. **Ледник (cold storage)** — *оригинальные* (до сушки) сообщения пользователя
   пишутся в `cold_storage.jsonl` (`log_to_cold_storage`), так что ничего не
   теряется, даже если из горячего контекста текст удалён.
6. **Спасательный круг** (`my_after_model_callback` → `_handle_empty_response`) —
   при пустом ответе или сбое API в ответ подставляется последнее сообщение
   пользователя, а счётчик сбоев растёт в `state["error_count"]`.
7. **Переключение провайдеров** (`get_llm_model`) — по переменной `AI_PROVIDER`:
   - `yandex` (по умолчанию) — Yandex AI Studio (`YANDEX_API_KEY`, `YANDEX_FOLDER_ID`, `YANDEX_API_BASE`);
   - `selectel` — Selectel AI Gateway (`SELECTEL_API_KEY`).

## Структура проекта

```
my_test_agent/
├── agent.py                # сам агент: колбэки, окно, сушка, ледник, провайдеры
├── measure_tokens_local.py # локальный замер токенов (DeepSeek-токенизатор + tiktoken)
├── __init__.py
├── .env                    # ключи провайдеров (не в git)
├── .adk/                   # сессии ADK (session.db)
├── cold_storage-Selectel.zip
└── docs/
    ├── README.md                                  # заметки по архитектуре
    ├── Тесты для Агента со скользящим окном.md    # сценарии тестов
    ├── Финальный отчёт (Selectel).md              # результаты эксперимента
    ├── Как подключить ИИ-роутер Selectel.md
    ├── tool_calbacks.md
    ├── router-usage-summary_start.csv             # потребление токенов, старт
    └── Saving tokens (Selectel|Yandex)/           # логи и метрики экспериментов
```

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install google-adk litellm python-dotenv transformers torch tiktoken sentencepiece
```

Первый запуск скачает модель эмоций (~25 МБ) из HuggingFace.

## Настройка окружения

Создайте `.env` рядом с `agent.py`:

```dotenv
AI_PROVIDER=yandex

# Yandex AI Studio
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
YANDEX_API_BASE=https://llm.api.cloud.yandex.net/v1

# Selectel AI Gateway (если AI_PROVIDER=selectel)
SELECTEL_API_KEY=...
```

## Запуск

```bash
adk web           # веб-интерфейс ADK Dev UI
adk run my_test_agent   # или интерактивный чат в терминале
```

Локальный замер токенов (сравнение «обычный / разреженный / предикаты»):

```bash
python measure_tokens_local.py
```

## Как это работает при каждом ходе

```
сообщение пользователя
        │
        ▼
before_model_callback
  1. оригинал ──► cold_storage.jsonl (ледник)
  2. история сессии ──► скользящее окно (10 событий)
  3. текст окна ──► сушка (sparsify_text) + эмоция (state)
  4. [Tone hint] ──► в конец последнего user-сообщения
  5. llm_request.contents = новое окно
        │
        ▼
     LLM (статичная инструкция, кэшируемая)
        │
        ▼
after_model_callback
  • пусто/сбой → подставить последнее сообщение пользователя
  • успех → лог ответа + метрики (prompt/cached tokens), флаг снят
```

## Замечания

- `STRICT_MODE` объявлен, но пока не используется.
- Эмодзи-вывод и `print`-логирование — для наглядности эксперимента.
- Данные экспериментов (CSV, JSON, отчёты) лежат в `docs/`.
