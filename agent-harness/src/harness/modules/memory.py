import json
import logging
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryProvider(ABC):
    """Базовый интерфейс для любого RAG или базы знаний."""
    
    @abstractmethod
    def search(self, query: str, limit: int = 3) -> list[str]:
        """Ищет релевантный контекст по запросу."""
        pass


class FileMemoryProvider(MemoryProvider):
    """
    Локальная заглушка RAG. Ищет по ключевым словам в cold_storage.jsonl.
    Не требует внешних баз данных, работает из коробки.
    """
    def __init__(self, jsonl_path: str):
        self.jsonl_path = Path(jsonl_path)

    def search(self, query: str, limit: int = 3) -> list[str]:
        if not self.jsonl_path.exists():
            return []
            
        query_words = set(query.lower().split())
        results = []
        
        try:
            with open(self.jsonl_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        text = entry.get("text", "")
                        if not text:
                            continue
                        
                        # Простейший скоринг: количество совпавших слов
                        text_words = set(text.lower().split())
                        score = len(query_words & text_words)
                        if score > 0:
                            results.append((score, text))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ошибка чтения базы памяти: {e}")
            return []
        
        # Сортируем по убыванию релевантности и обрезаем
        results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in results[:limit]]


class HttpMemoryProvider(MemoryProvider):
    """
    Адаптер для подключения к внешнему RAG (например, Qdrant, YDB, LangChain).
    Ожидает POST-запрос с JSON {"query": "...", "limit": N} и ответ {"context": ["..."]}
    """
    def __init__(self, url: str, api_key: str = ""):
        self.url = url
        self.api_key = api_key

    def search(self, query: str, limit: int = 3) -> list[str]:
        try:
            data = json.dumps({"query": query, "limit": limit}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            req = urllib.request.Request(self.url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                res = json.loads(response.read().decode("utf-8"))
                return res.get("context", [])
        except Exception as e:
            logger.warning(f"HTTP RAG недоступен: {e}")
            return []