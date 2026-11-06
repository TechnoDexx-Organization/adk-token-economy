import hashlib
import json
import os
from datetime import datetime, timezone


class ColdStorage:
    """
    Ледник: append-only журнал всего, что прошло через горячий контекст.

    Пишет **до** любой необратимой обработки (сушка), поэтому из него всегда
    можно восстановить оригинал. Каждая запись несёт `event_id`, `role` и
    `invocation_id`, чтобы её можно было однозначно сопоставить с событием ADK.
    """

    def __init__(self, base_dir: str):
        self.file_path = os.path.join(base_dir, "cold_storage.jsonl")

    def log(
        self,
        session_id: str,
        role: str,
        text: str,
        *,
        invocation_id: str = "",
        event_id: str = "",
        kind: str = "message",
    ) -> None:
        """Сохраняет одну реплику в ледник (оригинал, до обработки)."""
        entry = {
            "session_id": session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "role": role,
            "invocation_id": invocation_id,
            "event_id": event_id,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
        }
        self._append(entry)

    def log_manifest(
        self,
        session_id: str,
        *,
        event_id: str,
        invocation_id: str,
        role: str,
        original: str,
        processed: str,
        removed: list[dict],
        ops: list[str],
    ) -> None:
        """
        Манифест обратимости: что именно сделала обработка с текстом.

        `removed` — список вырезанных фрагментов: {"start", "end", "text"} в
        координатах оригинала. Применение его задом наперёд восстанавливает
        исходный текст точь-в-точь, независимо от того, что ещё случилось.
        """
        entry = {
            "session_id": session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "manifest",
            "role": role,
            "invocation_id": invocation_id,
            "event_id": event_id,
            "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "processed_sha256": hashlib.sha256(processed.encode("utf-8")).hexdigest(),
            "original_length": len(original),
            "processed_length": len(processed),
            "ops": ops,
            "removed": removed,
            "original": original,
            "processed": processed,
        }
        self._append(entry)

    def _append(self, entry: dict) -> None:
        try:
            directory = os.path.dirname(self.file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Ошибка записи в ледник: {e}")
