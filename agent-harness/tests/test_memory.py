"""Память: поиск по ключевым словам и кэш индекса с инвалидацией по mtime (P2-23)."""

import json
import os
import time

from harness.modules.memory import FactMemoryProvider, FileMemoryProvider


def write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_file_memory_search_by_keywords(tmp_path):
    path = tmp_path / "cold_storage.jsonl"
    write_jsonl(
        path,
        [
            {"text": "пользователь любит python и агентов"},
            {"text": "сегодня хорошая погода"},
            {"text": "python агент читает файлы"},
        ],
    )
    provider = FileMemoryProvider(str(path))

    results = provider.search("python агент", limit=2)

    assert len(results) == 2
    # обе записи содержат оба слова — порядок по убыванию скора, затем по позиции
    assert set(results) == {"пользователь любит python и агентов", "python агент читает файлы"}


def test_file_memory_missing_file_returns_empty(tmp_path):
    provider = FileMemoryProvider(str(tmp_path / "нет_такого.jsonl"))
    assert provider.search("что угодно") == []


def test_file_memory_cache_reloads_on_file_change(tmp_path):
    path = tmp_path / "cold_storage.jsonl"
    write_jsonl(path, [{"text": "старые воспоминания"}])
    provider = FileMemoryProvider(str(path))

    assert provider.search("старые") == ["старые воспоминания"]

    time.sleep(0.01)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": "новые воспоминания"}, ensure_ascii=False) + "\n")
    # mtime на некоторых ФС тикает раз в секунду — подстрахуемся явным сдвигом
    stat = os.stat(path)
    os.utime(path, (stat.st_atime + 2, stat.st_mtime + 2))

    assert provider.search("новые") == ["новые воспоминания"]


def test_fact_memory_search_and_limit(tmp_path):
    path = tmp_path / "facts.jsonl"
    provider = FactMemoryProvider(str(path))
    provider.add_fact("пользователь работает над agent-harness")
    provider.add_fact("пользователь любит кофе")
    provider.add_fact("нейтральный факт")

    results = provider.search("пользователь кофе", limit=5)

    assert results[0] == "пользователь любит кофе"
    assert "пользователь работает над agent-harness" in results


def test_fact_memory_cache_picks_up_new_facts(tmp_path):
    path = tmp_path / "facts.jsonl"
    provider = FactMemoryProvider(str(path))
    provider.add_fact("первый факт")
    assert provider.search("первый") == ["первый факт"]

    provider.add_fact("второй факт")

    assert provider.search("второй") == ["второй факт"]
