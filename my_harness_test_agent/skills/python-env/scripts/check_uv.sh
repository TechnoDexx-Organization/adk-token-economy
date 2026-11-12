#!/usr/bin/env bash
# check_uv.sh — безопасная проверка окружения Python/uv

echo "=== Проверка окружения ==="

# Проверяем наличие uv
if command -v uv >/dev/null 2>&1; then
    echo "UV: Установлен ($(uv --version))"
else
    echo "UV: НЕ УСТАНОВЛЕН. Необходимо установить."
fi

# Проверяем, активировано ли виртуальное окружение
if [ -n "$VIRTUAL_ENV" ]; then
    echo "VENV: Активно ($VIRTUAL_ENV)"
else
    echo "VENV: Не активировано."
fi

# Проверяем версию Python
if command -v python >/dev/null 2>&1; then
    echo "PYTHON: $(python --version 2>&1)"
else
    echo "PYTHON: Не найден в PATH."
fi