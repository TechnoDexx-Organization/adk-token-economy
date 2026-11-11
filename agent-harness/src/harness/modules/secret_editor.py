import logging
import math
import re
from collections import Counter

logger = logging.getLogger(__name__)


class SecretEditor:
    """
    Модуль безопасности (DLP), который вычищает потенциальные секреты
    из текста перед отправкой в LLM.
    """

    def __init__(self):
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
            # Классический UUID
            re.compile(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
            ),
        ]

    def _calculate_entropy(self, s: str) -> float:
        if not s:
            return 0.0
        counts = Counter(s)
        length = len(s)
        return -sum(
            (count / length) * math.log2(count / length) for count in counts.values()
        )

    def _is_secret_word(self, word: str) -> bool:
        # 1. Проверка по регуляркам (высокий приоритет)
        for pattern in self.secret_patterns:
            if pattern.search(word):
                return True

        # 2. Проверка белого списка (UUID)
        for pattern in self.allow_patterns:
            if pattern.match(word):
                return False

        # Очищаем слово от знаков препинания для анализа
        clean_word = re.sub(r"[^\w\-]", "", word)

        # 3. Фильтр по длине (короткие слова не могут быть ключами)
        if len(clean_word) < 20:
            return False

        # 4. Анализ алфавита и энтропии
        if re.match(r"^[0-9a-fA-F]+$", clean_word):
            # Это Hex (хэш SHA, MD5). Максимальная энтропия тут 4.0
            return self._calculate_entropy(clean_word) > 3.7
        elif re.match(r"^[a-zA-Z0-9\-_]+$", clean_word):
            # Это Base64 или похожий ключ
            return self._calculate_entropy(clean_word) > 4.2

        # Если там кириллица, спецсимволы или пробелы — это текст
        return False

    def redact(self, text: str) -> str:
        """
        Проходит по тексту, находит секреты и заменяет их на [REDACTED].
        """
        if not text:
            return text

        # Разбиваем текст по пробелам и переносам строк, сохраняя их (в скобках)
        words = re.split(r"(\s+)", text)
        safe_words = []

        for word in words:
            # Если это просто пробел или перенос строки — оставляем как есть
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

        # Склеиваем обратно без добавления лишних пробелов
        return "".join(safe_words)