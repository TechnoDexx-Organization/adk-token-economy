"""Cost & Token Tracking: учёт токенов и стоимости в реальном времени.

Харнесс считает токены и стоимость каждого вызова модели.
Если лимит исчерпан — бросает `CostLimitExceeded` и пользователь видит
сообщение об остановке.

Прайс-листы (₽ за 1M токенов):
  - DeepSeek R1:    input 0.14, cached 0.07, output 0.55
  - DeepSeek V3:    input 0.27, cached 0.07, output 1.10
  - GPT-4o:         input 2.50, cached 1.25, output 10.00
  - GPT-4o-mini:    input 0.15, cached 0.075, output 0.60

Параметры берутся из переменных окружения:
  - HARNESS_COST_LIMIT        — бюджет в рублях (например "5.00");
                                None или пустая строка => без лимита.
  - HARNESS_MODEL_PRICING     — JSON с кастомным прайсом,
                                {"model_name": {"input": X, "cached": Y, "output": Z}}.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass

__all__ = ["CostTracker", "CostLimitExceeded"]

logger = logging.getLogger(__name__)


class CostLimitExceeded(Exception):
    """Лимит стоимости исчерпан. Харнесс должен остановиться."""

    def __init__(self, spent: float, limit: float):
        self.spent = spent
        self.limit = limit
        super().__init__(f"Лимит стоимости исчерпан: потрачено {spent:.4f}₽ из {limit:.4f}₽")


# Прайс-листы по умолчанию (₽ за 1M токенов)
DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "deepseek-r1": {"input": 0.14, "cached": 0.07, "output": 0.55},
    "deepseek-r1-0528": {"input": 0.14, "cached": 0.07, "output": 0.55},
    "deepseek-v3": {"input": 0.27, "cached": 0.07, "output": 1.10},
    "deepseek-v3-0324": {"input": 0.27, "cached": 0.07, "output": 1.10},
    "gpt-4o": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-2024-11-20": {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached": 0.075, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "cached": 0.075, "output": 0.60},
}


@dataclass
class _ModelPricing:
    """Привязанный к модели прайс."""

    model_name: str
    input: float  # ₽ за 1M input токенов
    cached: float  # ₽ за 1M cached токенов
    output: float  # ₽ за 1M output токенов


class CostTracker:
    """
    Атомарный счётчик токенов и стоимости.

    Поток-безопасен: все мутации защищены Lock.
    """

    def __init__(
        self,
        limit: float | None = None,
        pricing: dict[str, dict[str, float]] | None = None,
    ):
        """
        :param limit: бюджет в рублях (None => без лимита).
        :param pricing: кастомный прайс-лист. Если не передан — используется
                        DEFAULT_PRICING + чтение из HARNESS_MODEL_PRICING.
        """
        self._limit = limit
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._lock = threading.Lock()

        # Читаем лимит из env
        env_limit = os.environ.get("HARNESS_COST_LIMIT", "").strip()
        if env_limit and self._limit is None:
            try:
                self._limit = float(env_limit)
            except ValueError:
                logger.warning("HARNESS_COST_LIMIT — невалидное число, игнорируем")

        # Собираем прайс-лист
        all_pricing: dict[str, dict[str, float]] = dict(DEFAULT_PRICING)
        # Кастомный прайс из параметра (приоритет над DEFAULT_PRICING)
        if pricing and isinstance(pricing, dict):
            all_pricing.update(pricing)
        # Кастомный прайс из env (приоритет над параметром)
        custom_json = os.environ.get("HARNESS_MODEL_PRICING", "").strip()
        if custom_json:
            try:
                custom = json.loads(custom_json)
                if isinstance(custom, dict):
                    all_pricing.update(custom)
            except json.JSONDecodeError:
                logger.warning("HARNESS_MODEL_PRICING — невалидный JSON, игнорируем")
        self._pricing_map: dict[str, _ModelPricing] = {}
        for model_key, prices in all_pricing.items():
            if isinstance(prices, dict) and "input" in prices:
                self._pricing_map[model_key.lower()] = _ModelPricing(
                    model_name=model_key,
                    input=float(prices["input"]),
                    cached=float(prices.get("cached", prices["input"])),
                    output=float(prices.get("output", prices["input"])),
                )

        # Кэш последней найденной модели (для быстрого пути)
        self._cached_pricing: _ModelPricing | None = None
        self._cached_key: str = ""

    def _resolve_pricing(self, model_name: str | None) -> _ModelPricing | None:
        """Ищет прайс по имени модели (регистронезависимо)."""
        if not model_name:
            return None
        key = model_name.lower()
        if self._cached_key == key and self._cached_pricing is not None:
            return self._cached_pricing
        pricing = self._pricing_map.get(key)
        if pricing is None:
            # Fallback: частичное совпадение
            for k, p in self._pricing_map.items():
                if k in key or key in k:
                    pricing = p
                    break
        if pricing is not None:
            self._cached_key = key
            self._cached_pricing = pricing
        return pricing

    def track(
        self,
        model_name: str | None,
        prompt_tokens: int,
        cached_tokens: int,
        total_tokens: int,
    ) -> dict:
        """
        Засчитать один вызов модели.

        :return: словарь с метриками для логирования.
        """
        pricing = self._resolve_pricing(model_name)

        # Если прайс не найден — считаем по средней ставке (input = output)
        if pricing is None:
            pricing = _ModelPricing(
                model_name=model_name or "unknown",
                input=1.0,
                cached=0.5,
                output=1.0,
            )
            logger.debug(
                "Прайс для модели '%s' не найден, используем fallback (1₽/M токенов)",
                model_name,
            )

        # output токены = total - prompt - cached (может быть 0 или отрицательным
        # из-за особенностей разных провайдеров)
        output_tokens = max(0, total_tokens - prompt_tokens - cached_tokens)

        cost = (
            (prompt_tokens - cached_tokens) * pricing.input / 1_000_000
            + cached_tokens * pricing.cached / 1_000_000
            + output_tokens * pricing.output / 1_000_000
        )

        with self._lock:
            self._total_tokens += total_tokens
            self._total_cost += cost

        # Логирование
        logger.info(
            "Cost: model=%s prompt=%d cached=%d output=%d total=%d cost=%.6f₽ total_cost=%.4f₽ total_tokens=%d",
            model_name or "?",
            prompt_tokens,
            cached_tokens,
            output_tokens,
            total_tokens,
            cost,
            self._total_cost,
            self._total_tokens,
            extra={
                "kind": "cost_tracking",
                "model": model_name,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_rub": round(cost, 6),
                "total_cost_rub": round(self._total_cost, 4),
                "total_tokens_all": self._total_tokens,
            },
        )

        return {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_rub": round(cost, 6),
            "total_cost_rub": round(self._total_cost, 4),
            "total_tokens_all": self._total_tokens,
        }

    def check_limit(self) -> None:
        """
        Проверяет, не превышен ли лимит.

        :raises CostLimitExceeded: если лимит исчерпан.
        """
        if self._limit is None:
            return
        with self._lock:
            if self._total_cost >= self._limit:
                raise CostLimitExceeded(spent=self._total_cost, limit=self._limit)

    @property
    def total_cost_rub(self) -> float:
        """Текущая общая стоимость в рублях (thread-safe)."""
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Общее количество токенов (thread-safe)."""
        with self._lock:
            return self._total_tokens

    @property
    def limit(self) -> float | None:
        return self._limit

    def reset(self) -> None:
        """Сбросить счётчики (для тестов)."""
        with self._lock:
            self._total_tokens = 0
            self._total_cost = 0.0
            self._cached_pricing = None
            self._cached_key = ""
