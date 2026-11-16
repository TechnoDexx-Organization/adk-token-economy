"""Тесты DLP: аудит редактирования секретов (P0-8)."""
import hashlib
import json

from harness.modules.secret_editor import SecretEditor
from harness.modules.storage import ColdStorage


def test_redaction_is_logged_to_cold_storage(tmp_path):
    """
    P0-8 (фикс): SecretEditor вырезает секрет и пишет в аудит.
    Тест проверяет, что создается запись kind='redaction' с sha256,
    но без самого секрета.
    """
    storage = ColdStorage(str(tmp_path))
    editor = SecretEditor(storage=storage)
    
    secret = "sk-1234567890abcdefghijklmnop"
    text = f"Here is my key: {secret}"
    
    # Вызываем redact с контекстом, чтобы знать, куда писать
    redacted = editor.redact(
        text, 
        session_id="s1", 
        event_id="e1", 
        invocation_id="inv1", 
        role="user"
    )
    
    assert secret not in redacted
    assert "[REDACTED]" in redacted
    
    # Проверяем, что в леднике есть запись
    with open(tmp_path / "cold_storage.jsonl", encoding="utf-8") as f:
        lines = f.readlines()
    
    assert len(lines) == 1
    entry = json.loads(lines[0])
    
    assert entry["kind"] == "redaction"
    assert entry["session_id"] == "s1"
    assert entry["event_id"] == "e1"
    assert entry["sha256"] == hashlib.sha256(secret.encode("utf-8")).hexdigest()
    assert "text" not in entry  # Самое важное: мы не храним пароль в открытом виде!