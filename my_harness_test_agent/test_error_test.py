"""Тесты для error_test.py: скрипт должен выводить ровно "Ошибка: деление на ноль"."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "error_test.py"
EXPECTED = "Ошибка: деление на ноль"


def _run_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"Скрипт не найден: {SCRIPT}"


def test_output_is_exact_error_message() -> None:
    result = _run_script()
    assert result.stdout.strip() == EXPECTED, (
        f"Ожидалось {EXPECTED!r}, получено {result.stdout.strip()!r} "
        f"(stderr: {result.stderr.strip()!r})"
    )


def test_no_stderr_output() -> None:
    result = _run_script()
    assert result.stderr == "", f"В stderr есть вывод: {result.stderr!r}"


def test_exit_code_is_zero() -> None:
    result = _run_script()
    assert result.returncode == 0, (
        f"Код возврата {result.returncode}, stderr: {result.stderr.strip()!r}"
    )
