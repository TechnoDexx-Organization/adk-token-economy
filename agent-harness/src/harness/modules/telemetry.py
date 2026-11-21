"""Structured Logging: JSON-логи и опциональная поставка в Monium (OTLP).

Библиотека по-прежнему не настраивает логирование за приложение (NullHandler).
`configure_logging()` — это **явный** вызов из точки запуска приложения:

    from harness import configure_logging
    configure_logging()

Параметры берутся из переменных окружения (конвенция OTel, Monium их понимает):
  - OTEL_SERVICE_NAME          — имя сервиса (атрибут ресурса `service`);
  - OTEL_EXPORTER_OTLP_ENDPOINT— адрес OTLP (для Monium: https://ingest.monium.yandex.cloud:443);
  - OTEL_RESOURCE_ATTRIBUTES   — дополнительные пары key=value (cluster, deployment.name...);
  - HARNESS_LOG_FORMAT         — "json" (по умолчанию) или "text".

Если установлены пакеты extra `observability` и задан OTLP-endpoint — логи
уходят OTLP-экспортёром OpenTelemetry (с trace_id/span_id через
opentelemetry-instrumentation-logging). Иначе — JSON-строки в stdout,
по одной записи на событие, со всеми extra-полями.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

__all__ = ["JsonFormatter", "configure_logging"]

# служебный флаг: помечает хендлеры, созданные configure_logging
_MANAGED_FLAG = "_harness_managed"

# служебные атрибуты LogRecord, которые не относятся к структуре лога
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """Один лог = одна JSON-строка. Поля из `extra` попадают в лог как есть."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _attach_otel_handler() -> logging.Handler | None:
    """OTLP-хендлер, если extra `observability` установлен и endpoint задан."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return None
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        logging.getLogger(__name__).warning(
            "OTLP endpoint задан, но пакеты extra 'observability' не установлены — "
            "логи останутся в stdout (pip install agent-harness[observability])"
        )
        return None

    resource_attrs = {"service.name": os.environ.get("OTEL_SERVICE_NAME", "agent-harness")}
    for pair in os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "").split(","):
        if "=" in pair:
            key, _, value = pair.partition("=")
            resource_attrs[key.strip()] = value.strip()

    provider = LoggerProvider(resource=Resource.create(resource_attrs))
    provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter(endpoint=f"{endpoint}/v1/logs")))
    set_logger_provider(provider)
    return LoggingHandler(logger_provider=provider)


def configure_logging(level: int | str = logging.INFO) -> logging.Handler:
    """
    Включает структурированное логирование для всего приложения.

    Идемпотентна: повторный вызов заменяет хендлер, а не дублирует его.
    Возвращает установленный хендлер (для тестов и отладки).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # снять предыдущий хендлер configure_logging, чтобы не дублировать строки
    for handler in list(root.handlers):
        if getattr(handler, _MANAGED_FLAG, False):
            root.removeHandler(handler)

    handler = _attach_otel_handler()
    if handler is not None:
        handler.setLevel(level)
        # служебный флаг: помечает хендлер, которым управляет configure_logging
        setattr(handler, _MANAGED_FLAG, True)  # noqa: B010
        root.addHandler(handler)
        return handler

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.setLevel(level)
    setattr(handler, _MANAGED_FLAG, True)  # noqa: B010
    root.addHandler(handler)
    return handler
