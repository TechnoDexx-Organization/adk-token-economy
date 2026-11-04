import json
import os
from datetime import datetime, timezone

class ColdStorage:
    def __init__(self, base_dir: str):
        self.file_path = os.path.join(base_dir, "cold_storage.jsonl")

    def log(self, session_id: str, role: str, text: str):
        entry = {
            "session_id": session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "text": text,
        }
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Ошибка записи в ледник: {e}")
