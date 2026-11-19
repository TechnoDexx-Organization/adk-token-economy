"""Тесты честной обратимости (P0-3). Проверяют восстановление после replace."""

import json

from harness.adapters.adk import ADKAgentCallbacks
from harness.modules.sparsifier import TextSparsifier
from harness.modules.storage import ColdStorage
from harness.modules.window import SlidingWindow


def test_replace_operation_is_reversible(tmp_path):
    """
    P0-3 (фикс): Реальный TextSparsifier делает замену " потому что " → ". ".
    Теперь мы используем opcodes, и восстановление проходит идеально.
    """
    original = "Я устал, потому что долго работал"
    sparsifier = TextSparsifier()
    processed = sparsifier.process(original, {})

    assert "потому что" not in processed
    assert "." in processed

    storage = ColdStorage(str(tmp_path))
    removed_spans = ADKAgentCallbacks._get_opcodes(original, processed)

    storage.log_manifest(
        "session-1",
        event_id="e1",
        invocation_id="inv1",
        role="user",
        original=original,
        processed=processed,
        opcodes=removed_spans,
        ops=["sparsify"],
    )

    with open(tmp_path / "cold_storage.jsonl", encoding="utf-8") as f:
        manifest = json.loads(f.readline())

    # НОВЫЙ АЛГОРИТМ: используем встроенный restore
    restored = ColdStorage.restore(manifest)

    assert restored == original, f"Ожидалось: {original!r}\nПолучено: {restored!r}"


def test_cap_manifest_has_text_for_restore(tmp_path):
    """
    P0-3 (фикс): Cap-манифест теперь содержит opcodes, поэтому восстанавливается.
    """
    long_text = "очень длинный текст " * 600
    window = SlidingWindow(max_messages=10, max_message_tokens=200)
    compacted = window.compact(long_text)

    storage = ColdStorage(str(tmp_path))
    callbacks = ADKAgentCallbacks(storage=storage, window=window)

    class FakeEvent:
        id = "e1"
        invocation_id = "inv1"
        author = "user"
        content = None

    callbacks._log_cap_manifest("session-1", FakeEvent(), long_text, compacted)

    with open(tmp_path / "cold_storage.jsonl", encoding="utf-8") as f:
        manifest = json.loads(f.readline())

    # Проверяем, что восстановление работает
    restored = ColdStorage.restore(manifest)
    assert restored == long_text
