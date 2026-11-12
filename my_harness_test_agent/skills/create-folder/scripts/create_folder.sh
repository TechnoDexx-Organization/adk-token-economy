#!/usr/bin/env bash
if [ -z "$1" ]; then
    echo "Ошибка: Укажите имя папки. Пример: ./create_folder.sh my_folder"
    exit 1
fi

mkdir -p "$1"
echo "✅ Папка '$1' создана."
