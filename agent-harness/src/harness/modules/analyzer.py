"""Анализ эмоциональной тональности сообщений."""

import logging

logger = logging.getLogger(__name__)

EMOTION_MODEL = "Aniemore/rubert-tiny2-russian-emotion-detection"


class EmotionAnalyzer:
    """
    Классификатор тональности на базе rubert-tiny2.

    Модель загружается лениво — при первом вызове analyze(), а не в конструкторе
    (P2-21): импорт и создание анализатора больше не тянут transformers/torch и
    не блокируют запуск агента. Результат по тексту кэшируется, повторные
    запросы не обращаются к модели.
    """

    def __init__(self, model_name: str = EMOTION_MODEL, cache_size: int = 1024):
        self.model_name = model_name
        self.cache_size = cache_size
        self._pipeline = None
        self._load_failed = False
        self._cache: dict[str, str] = {}

    @property
    def pipeline(self):
        """Ленивая загрузка модели: None при недоступности transformers."""
        if self._pipeline is None and not self._load_failed:
            try:
                from transformers import pipeline

                logger.info("Загрузка модели эмоционального анализа: %s", self.model_name)
                self._pipeline = pipeline("text-classification", model=self.model_name)
                logger.info("Модель эмоционального анализа загружена")
            except Exception as e:  # noqa: BLE001 — падение модели не должно ронять агента
                self._load_failed = True
                logger.warning("Не удалось загрузить модель: %s", e)
        return self._pipeline

    def analyze(self, text: str) -> str:
        if not text:
            return "neutral"
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        result = "neutral"
        if pipe := self.pipeline:
            try:
                result = pipe(text, truncation=True, max_length=512)[0]["label"]
            except Exception:  # noqa: BLE001 — на любой сбой классификатора отвечаем нейтрально
                result = "neutral"
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[text] = result
        return result
