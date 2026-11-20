import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from .ssrf import is_safe_url

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Базовый интерфейс для любого RAG или базы знаний."""

    @abstractmethod
    def search(self, query: str, limit: int = 3) -> list[str]:
        """Ищет релевантный контекст по запросу."""
        pass


class _JsonlKeywordIndex:
    """
    Кэш JSONL-файла для поиска по ключевым словам (P2-23).

    Файл читается целиком только при первом запросе или когда изменился его
    mtime (записи добавляются append-ом). Индексируется только содержимое,
    помеченное `field`; битые строки молча пропускаются.
    """

    def __init__(self, jsonl_path: Path, field: str):
        self.jsonl_path = jsonl_path
        self.field = field
        self._entries: list[tuple[int, frozenset[str]]] | None = None
        self._mtime: float | None = None

    def invalidate(self) -> None:
        """Сбрасывает кэш (вызывать после гарантированной записи в файл)."""
        self._entries = None
        self._mtime = None

    def _load(self) -> list[tuple[int, frozenset[str]]]:
        try:
            mtime = self.jsonl_path.stat().st_mtime
        except OSError:
            self.invalidate()
            return []
        if self._entries is not None and mtime == self._mtime:
            return self._entries

        entries: list[tuple[int, frozenset[str]]] = []
        try:
            with open(self.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = entry.get(self.field, "")
                    if not text:
                        continue
                    entries.append((len(entries), frozenset(text.lower().split())))
        except Exception as e:  # noqa: BLE001
            logger.error(f"Ошибка чтения базы памяти: {e}")
            return []

        self._entries = entries
        self._mtime = mtime
        return entries

    def search(self, query: str, limit: int, read_text) -> list[str]:
        """Возвращает тексты записей, отсортированные по пересечению слов."""
        if not self.jsonl_path.exists():
            return []
        entries = self._load()
        if not entries:
            return []

        query_words = set(query.lower().split())
        scored = [(len(query_words & words), index) for index, words in entries if (query_words & words)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [read_text(index) for _, index in scored[:limit]]


class FileMemoryProvider(MemoryProvider):
    """
    Локальная заглушка RAG. Ищет по ключевым словам в cold_storage.jsonl.
    Не требует внешних баз данных, работает из коробки.
    """

    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self._index = _JsonlKeywordIndex(self.jsonl_path, field="text")

    def _text_at(self, index: int) -> str:
        with open(self.jsonl_path, encoding="utf-8") as f:
            for position, line in enumerate(f):
                if position != index:
                    continue
                try:
                    return json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    return ""
        return ""

    def search(self, query: str, limit: int = 3) -> list[str]:
        return self._index.search(query, limit, self._text_at)


class FactMemoryProvider:
    """Память фактов о пользователе. Хранит структурированные записи в JSONL."""

    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._index = _JsonlKeywordIndex(self.jsonl_path, field="fact")

    def add_fact(self, fact: str, source: str = "user") -> None:
        """Добавляет факт в память."""
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "fact": fact.strip(),
            "source": source,
        }
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Файл изменился — индекс должен перечитаться на следующем запросе.
        self._index.invalidate()

    def _fact_at(self, index: int) -> str:
        with open(self.jsonl_path, encoding="utf-8") as f:
            for position, line in enumerate(f):
                if position != index:
                    continue
                try:
                    return json.loads(line).get("fact", "")
                except json.JSONDecodeError:
                    return ""
        return ""

    def search(self, query: str, limit: int = 5) -> list[str]:
        """Ищет факты по ключевым словам (простой скоринг)."""
        return self._index.search(query, limit, self._fact_at)


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """SSRF-обход через redirect запрещён: 3xx считается ошибкой."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, f"SSRF: redirect запрещён -> {newurl}", headers, fp)


class HttpMemoryProvider(MemoryProvider):
    """
    Адаптер для подключения к внешнему RAG (например, Qdrant, YDB, LangChain).
    Ожидает POST-запрос с JSON {"query": "...", "limit": N} и ответ {"context": ["..."]}

    Защита от SSRF: URL проверяется до запроса (localhost, приватные и
    служебные сети), редиректы не переходятся.
    """

    def __init__(self, url: str, api_key: str = ""):
        if not is_safe_url(url):
            raise ValueError(f"SSRF: небезопасный URL памяти: {url}")
        self.url = url
        self.api_key = api_key
        # Отдельный opener: redirect на внутренний адрес не переходится.
        self._opener = urllib.request.build_opener(_NoRedirectHandler)

    def search(self, query: str, limit: int = 3) -> list[str]:
        if not is_safe_url(self.url):
            # Хост мог сменить резолв с момента конструктора — проверяем снова.
            logger.warning(
                "SSRF: URL памяти стал небезопасным, запрос отклонён: %s",
                self.url,
                extra={"kind": "ssrf_block", "provider": "HttpMemoryProvider", "url": self.url},
            )
            return []
        try:
            data = json.dumps({"query": query, "limit": limit}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            req = urllib.request.Request(self.url, data=data, headers=headers)
            with self._opener.open(req, timeout=5) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("context", [])
        except Exception as e:  # noqa: BLE001
            logger.warning(f"HTTP RAG недоступен: {e}")
            return []
