"""Тесты Structured Logging: JSON-формат и настройка из переменных окружения."""

import json
import logging

import pytest

from harness.modules.telemetry import JsonFormatter, configure_logging


def make_record(**extra):
    record = logging.LogRecord(
        name="harness.modules.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="SSRF: запрос отклонён %s",
        args=("/internal",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_emits_one_valid_json_line():
    line = JsonFormatter().format(make_record())
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "harness.modules.test"
    assert payload["message"] == "SSRF: запрос отклонён /internal"
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_includes_extra_fields():
    payload = json.loads(
        JsonFormatter().format(
            make_record(session_id="s1", event_id="e7", tool="run_shell_command", kind="ssrf_block", count=3)
        )
    )
    assert payload["session_id"] == "s1"
    assert payload["event_id"] == "e7"
    assert payload["tool"] == "run_shell_command"
    assert payload["kind"] == "ssrf_block"
    assert payload["count"] == 3


def test_json_formatter_keeps_non_ascii_readable():
    line = JsonFormatter().format(logging.LogRecord("t", 20, "f", 1, "Привет мир", (), None))
    assert "Привет мир" in line  # ensure_ascii=False


def test_json_formatter_serializes_unserializable_via_str():
    class Weird:
        def __str__(self):
            return "weird!"

    payload = json.loads(JsonFormatter().format(make_record(payload_obj=Weird())))
    assert payload["payload_obj"] == "weird!"


def test_json_formatter_adds_exc_info(monkeypatch):
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = make_record()
        record.exc_info = sys.exc_info()
    payload = json.loads(formatter.format(record))
    assert "ValueError: boom" in payload["exc_info"]


@pytest.fixture
def clean_root():
    """Восстанавливает корневой логгер после теста."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_configure_logging_emits_json_to_stdout(clean_root, capsys):
    configure_logging(level=logging.INFO)
    logging.getLogger("harness.modules.guard").warning(
        "Команда заблокирована: %s", "rm -rf /", extra={"session_id": "s1", "kind": "guard_deny"}
    )
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["kind"] == "guard_deny"
    assert payload["session_id"] == "s1"


def test_configure_logging_is_idempotent(clean_root, capsys):
    configure_logging()
    configure_logging()  # повторный вызов не должен удваивать вывод
    logging.getLogger("x").error("один раз")
    out = capsys.readouterr().out.strip()
    assert out.count("один раз") == 1


def test_configure_logging_without_otel_endpoint_stays_on_stdout(clean_root, monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    handler = configure_logging()
    # OTLP нет — обычный StreamHandler с JsonFormatter
    assert not hasattr(handler, "exporter")
    assert type(handler.formatter).__name__ == "JsonFormatter"
