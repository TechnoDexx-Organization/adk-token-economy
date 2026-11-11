import logging
import math
import pathlib
import re
from collections import Counter

logger = logging.getLogger(__name__)


class SecretEditor:
    """
    Модуль безопасности (DLP), который вычищает потенциальные секреты
    из текста перед отправкой в LLM.
    """

    def __init__(self, allowlist_path: str | None = None):
        # Уровень 1: Регулярки (явные секреты)
        self.secret_patterns = [
            re.compile(r"sk-[a-zA-Z0-9\-_]{20,}"),  # OpenAI
            re.compile(r"eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+"),  # JWT
            re.compile(
                r"(?i)(?:api[_-]?key|password|token)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-]{10,})['\"]?"
            ),
        ]
        # Уровень 1.5: Белый список (что нельзя скрывать)
        self.allow_patterns = [
            re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
            ),
        ]
        # Механизм для динамического наполнения
        self.allowed_tokens: set[str] = set()
        
        # Если передан путь — загружаем из файла
        if allowlist_path:
            self.load_allowlist_from_file(allowlist_path)

    def add_allowed_token(self, token: str) -> None:
        """Добавляет конкретный токен в белый список."""
        if token:
            # Добавляем и оригинал, и очищенный вариант (без знаков препинания)
            clean = re.sub(r"[^\w\-]", "", token)
            self.allowed_tokens.add(token.strip())
            self.allowed_tokens.add(clean)
            logger.info(f"🔓 SecretEditor: добавлен в белый список токен (длина {len(token)})")

    def load_allowlist_from_file(self, path: str) -> None:
        """Читает файл со списком разрешенных токенов."""
        p = pathlib.Path(path)
        if not p.exists():
            logger.warning(f"⚠️ SecretEditor: файл allowlist не найден: {path}")
            return
            
        # Используем utf-8-sig, чтобы убрать невидимый BOM в начале файла
        with p.open("r", encoding="utf-8-sig") as f:
            for line in f:
                token = line.strip()
                # Игнорируем пустые строки и комментарии
                if token and not token.startswith("#"):
                    self.add_allowed_token(token)
        
        logger.info(f"📖 SecretEditor: загружено {len(self.allowed_tokens)} уникальных токенов из {path}")

    def _calculate_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        counts = Counter(s)
        length = len(s)
        return -sum(
            (count / length) * math.log2(count / length) for count in counts.values()
        )

    def _is_secret_word(self, word: str) -> bool:
        # Очищаем слово от знаков препинания (точки, запятые, кавычки)
        clean_word = re.sub(r"[^\w\-]", "", word)

        # 0. Проверка белого списка (сравниваем и оригинал, и очищенный вариант)
        if word in self.allowed_tokens or clean_word in self.allowed_tokens:
            return False

        # 1. Проверка по регуляркам
        for pattern in self.secret_patterns:
            if pattern.search(word):
                return True

        # 2. Проверка белого списка (UUID)
        for pattern in self.allow_patterns:
            if pattern.match(clean_word):
                return False

        # 3. Фильтр по длине
        if len(clean_word) < 20:
            return False

        # 4. Анализ алфавита и энтропии
        if re.match(r"^[0-9a-fA-F]+$", clean_word):
            return self._calculate_entropy(clean_word) > 3.7
        elif re.match(r"^[a-zA-Z0-9\-_]+$", clean_word):
            return self._calculate_entropy(clean_word) > 4.2

        return False

    def redact(self, text: str) -> str:
        if not text:
            return text

        words = re.split(r"(\s+)", text)
        safe_words = []

        for word in words:
            if word.isspace():
                safe_words.append(word)
                continue

            if self._is_secret_word(word):
                logger.warning(
                    f"🛡️ SecretEditor: скрыт потенциальный секрет (длина {len(word)})"
                )
                safe_words.append("[REDACTED]")
            else:
                safe_words.append(word)

        return "".join(safe_words)