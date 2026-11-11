# test_uuid_experiment.py
import math
import re
from collections import Counter

def calculate_entropy(s: str) -> float:
    if not s: return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())

class SmartSecretEditor:
    def __init__(self):
        # Уровень 1: Регулярки (Секреты)
        self.secret_patterns = [
            re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),        # OpenAI
            re.compile(r"eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+"), # JWT
            re.compile(r"(?i)(?:api[_-]?key|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{10,})['\"]?"),
        ]
        # Уровень 1.5: Исключения (UUID, Git Hashes - их не трогаем!)
        self.allow_patterns = [
            # Классический UUID
            re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
        ]

    def is_secret(self, word: str) -> bool:
        # 1. Проверка на секрет по регуляркам
        for pattern in self.secret_patterns:
            if pattern.search(word):
                return True

        # 2. Проверка на разрешенные штуки (UUID)
        for pattern in self.allow_patterns:
            if pattern.match(word):
                return False # Это UUID, не трогаем!

        # 3. Эвристика и Энтропия
        clean_word = re.sub(r'[^\w\-]', '', word)
        if len(clean_word) < 20:
            return False # Слишком короткое

        # Определяем тип строки (алфавит)
        if re.match(r"^[0-9a-fA-F]+$", clean_word):
            # Это Hex (хэш). Максимальная энтропия тут 4.0
            return calculate_entropy(clean_word) > 3.7
        elif re.match(r"^[a-zA-Z0-9\-_]+$", clean_word):
            # Это Base64 или похожий ключ
            return calculate_entropy(clean_word) > 4.2
        
        return False # Если там русские буквы или спецсимволы - это текст

# Тестируем
editor = SmartSecretEditor()
tests = {
    "UUID (не трогать)": "123e4567-e89b-12d3-a456-426614174000",
    "JWT (скрыть)": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ",
    "Hash SHA-256 (скрыть)": "a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3",
    "Русский текст (не трогать)": "приветкакделауменявсехорошоспасибоспокойнойночи",
}

for name, text in tests.items():
    result = "⚠️ СКРЫТЬ" if editor.is_secret(text) else "✅ ОСТАВИТЬ"
    print(f"{result}: {name}")
