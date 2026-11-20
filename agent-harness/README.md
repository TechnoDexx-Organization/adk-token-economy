# Agent Harness

Универсальный конструктор (обвязка) для любых LLM-агентов.
Позволяет независимо подключать сушку текста, скользящее окно, анализ эмоций, ледник (cold storage), память и защиту инструментов (ToolGuard).

## Модули

| Модуль | Что делает |
|---|---|
| `modules/window.py` | Окно контекста: по сообщениям, по ходам или по бюджету токенов + потолок на одно сообщение |
| `modules/tokens.py` | `TokenCounter` — оценка токенов на tiktoken, с фолбэком по символам |
| `modules/sparsifier.py` | Сушка: удаление слов-связок и дробление сложных предложений |
| `modules/analyzer.py` | `EmotionAnalyzer` — эмоция сообщения (rubert-tiny2), ленивая загрузка модели + кэш |
| `modules/storage.py` | `ColdStorage` — журнал всех реплик до обработки + манифест изменений (обратимость) + аудит DLP |
| `modules/secret_editor.py` | `SecretEditor` — DLP: вырезание секретов из текста с аудитом в ледник (по sha256) |
| `modules/guard.py` | `ShellCommandGuard` / `SafeToolGuard` — гибридный заслон опасным инструментам (`GuardDecision`: DENY → ALLOW → HITL) |
| `modules/memory.py` | `FileMemoryProvider` / `HttpMemoryProvider` / `FactMemoryProvider` — память с кэшем индекса |
| `modules/file_tools.py` | `FileTools` — чтение/запись файлов + `run_shell_command` под контролем Guard, Outbound DLP и SSRF-контроль вывода |
| `modules/ssrf.py` | `is_safe_url` / `find_blocked_urls` — заслон SSRF: localhost, приватные и служебные сети, DNS-rebinding, fail closed |
| `modules/canary.py` | `CanaryGuard` — канареечный токен в системном промпте + детекция утечки на выходе и провокации на входе |
| `modules/telemetry.py` | `JsonFormatter` + `configure_logging()` — JSON-логи в stdout + OTLP в Monium при `OTEL_EXPORTER_OTLP_ENDPOINT`; endpoint, `service.name` и ресурсы — из env |
| `modules/retry.py` | `build_retry_config` — ретраи обращений к LLM (litellm-совместимые kwargs) |
| `modules/skills.py` | `build_safe_skill_toolset` — динамические навыки из папки |
| `modules/llm_factory.py` | `build_llm_model` — фабрика LLM (Yandex, Selectel, Selectel Router) |
| `adapters/adk.py` | `ADKAgentCallbacks` — перевод модулей на колбэки Google ADK |

Публичный API доступен прямо из пакета: `from harness import SlidingWindow`.
Импорты ленивые (PEP 562): `import harness` не тянет `google-adk`, `litellm` и `transformers` — они подгружаются только при обращении к соответствующему символу.

## Контракт окна

`SlidingWindow` принимает один или несколько ограничителей. Они применяются по цепочке:

1. `max_messages` — оставить последние N сообщений;
2. `max_turns` — оставить последние N ходов (ход = один `invocation_id`);
3. `max_tokens` — выбросить самые старые, пока окно не влезет в бюджет;
4. `max_message_tokens` — сократить **одно** сообщение до головы и хвоста.

Пункты 1–3 решают, **какие** сообщения попадут в запрос; пункт 4 — **сколько** от каждого останется.
Именно пункт 4 даёт рост экономии вместе с размером сообщения: текст короче потолка не меняется
вообще, а длинный режется пропорционально своей длине.
Если ограничители не заданы вовсе, история сохраняется целиком (неявного `max_turns=10` нет).

## Обратимость

Любая необратимая обработка (сушка, усечение) пишет в ледник манифест с opcodes из
`difflib`. Восстановление оригинала: `ColdStorage.restore(manifest)` — текст совпадает точь-в-точь.

## Структурированное логирование (Structured Logging)

Ядро не настраивает логирование само — остаётся `NullHandler`. Вызов `configure_logging(level)`
из приложения:

- ставит `JsonFormatter` в корневой логгер (одна JSON-строка на запись; `extra`-поля
  (`kind`, `session_id`, `event_id`, `direction` и т.д.) попадают в JSON);
- при заданной переменной `OTEL_EXPORTER_OTLP_ENDPOINT` подключает OTLP-хендлер
  OpenTelemetry (`opentelemetry-exporter-otlp-proto-http`) — данные летят на
  `ingest.monium.yandex.cloud:443`;
- `service.name` и `cluster` / `deployment.name` берутся из `OTEL_SERVICE_NAME`
  и `OTEL_RESOURCE_ATTRIBUTES`;
- без extra-зависимостей OTel — JSON в stdout.

Установка: `pip install -e ".[observability]"`.

## Безопасность (Security Hardening)

- **SSRF.** `HttpMemoryProvider` проверяет URL до запроса: запрещены localhost и его алиасы,
  приватные/link-local/CGNAT-сети (включая метаданные облака `169.254.169.254`), не-HTTP схемы
  и хосты, которые резолвятся в такие адреса (DNS-rebinding). Ошибка DNS — отказ (fail closed).
  Редиректы не переходятся. Проверка дублируется перед каждым `search()`.
- **Outbound DLP.** `FileTools.write_file` прогоняет содержимое через `SecretEditor` до записи
  на диск (аудит — `kind="redaction"`, `role="outbound"`). В выводе `run_shell_command`
  подсвечиваются SSRF-адреса.
- **Canary Tokens.** `CanaryGuard` генерирует криптостойкий маркер на сессию
  (`canary_guard.marker` — вставить в системный промпт). Маркер в ответе модели —
  утечка промпта (`state["canary_leak"]=True`); маркер или его sha256 во входящем
  тексте — провокация, сообщение заменяется заглушкой `[BLOCKED...]`. В ледник
  пишется только факт хита, утёкший текст — нет.

## Тесты

```bash
pip install -e ".[dev]"
pytest                 # 83 тестов, сеть и API-ключи не нужны
```

pythonpath настроен в `pyproject.toml`, поэтому тесты идут и на чистом клоне без установки.

Покрыто: выбор окна по сообщениям/ходам/токенам и их цепочка, потолок на сообщение и рост
экономии с размером текста, полнота ледника, обратимость манифеста, обработка сообщения ровно
один раз, сохранение вызовов инструментов, блокировка опасных инструментов, HITL-подтверждения,
аудит DLP, ретраи, память и её кэш, SSRF (включая DNS-rebinding и fail closed),
Outbound DLP при записи файлов, канареечные токены на входе и выходе.
