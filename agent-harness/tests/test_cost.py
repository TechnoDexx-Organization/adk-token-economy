"""Тесты для agent-harness/src/harness/modules/cost.py"""

import json
from unittest.mock import patch

import pytest

from harness.modules.cost import DEFAULT_PRICING, CostLimitExceeded, CostTracker


class TestCostTrackerBasic:
    """Базовые тесты CostTracker."""

    def test_initial_state_no_limit(self):
        tracker = CostTracker()
        assert tracker.total_cost_rub == 0.0
        assert tracker.total_tokens == 0
        assert tracker.limit is None

    def test_initial_state_with_limit(self):
        tracker = CostTracker(limit=10.0)
        assert tracker.limit == 10.0

    def test_track_basic(self):
        tracker = CostTracker()
        tracker.track(model_name="deepseek-chat", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        assert tracker.total_tokens > 0
        assert tracker.total_cost_rub > 0.0

    def test_track_multiple_calls(self):
        tracker = CostTracker()
        tracker.track(model_name="deepseek-chat", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        first_cost = tracker.total_cost_rub
        first_tokens = tracker.total_tokens

        tracker.track(model_name="deepseek-chat", prompt_tokens=200, cached_tokens=100, total_tokens=500)
        assert tracker.total_cost_rub > first_cost
        assert tracker.total_tokens > first_tokens

    def test_reset(self):
        tracker = CostTracker()
        tracker.track(model_name="deepseek-chat", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        assert tracker.total_cost_rub > 0.0
        tracker.reset()
        assert tracker.total_cost_rub == 0.0
        assert tracker.total_tokens == 0


class TestCostCalculation:
    """Тесты расчёта стоимости."""

    def test_deepseek_r1_pricing(self):
        """Проверка формулы для deepseek-r1 с дефолтным прайсом."""
        tracker = CostTracker()
        # prompt=100, cached=50, total=300 => output=300-100-50=150
        tracker.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        # cost = (100-50)*0.14/1M + 50*0.07/1M + 150*0.55/1M
        expected = (50 * 0.14 + 50 * 0.07 + 150 * 0.55) / 1_000_000
        assert tracker.total_cost_rub == pytest.approx(expected, rel=1e-9)

    def test_cached_tokens_cheaper(self):
        """Кэшированные токены дешевле обычных."""
        t1 = CostTracker()
        t1.track(model_name="deepseek-r1", prompt_tokens=1000, cached_tokens=0, total_tokens=2000)

        t2 = CostTracker()
        t2.track(model_name="deepseek-r1", prompt_tokens=1000, cached_tokens=500, total_tokens=2000)
        assert t2.total_cost_rub < t1.total_cost_rub

    def test_unknown_model_fallback(self):
        """Неизвестная модель использует fallback-прайс."""
        tracker = CostTracker()
        tracker.track(model_name="unknown-model", prompt_tokens=1000, cached_tokens=0, total_tokens=2000)
        # fallback: input=1.0, cached=0.5, output=1.0
        expected = (1000 * 1.0 + 0 * 0.5 + 1000 * 1.0) / 1_000_000
        assert tracker.total_cost_rub == pytest.approx(expected, rel=1e-9)

    def test_zero_tokens(self):
        """Трек с нулевыми токенами не должен ломаться."""
        tracker = CostTracker()
        tracker.track(model_name="deepseek-r1", prompt_tokens=0, cached_tokens=0, total_tokens=0)
        assert tracker.total_cost_rub == 0.0


class TestCostLimit:
    """Тесты лимита стоимости."""

    def test_limit_exceeded(self):
        tracker = CostTracker(limit=0.001)
        tracker.track(model_name="deepseek-r1", prompt_tokens=1000000, cached_tokens=0, total_tokens=2000000)
        assert tracker.total_cost_rub >= 0.001

    def test_cost_limit_exceeded_exception(self):
        # Создаём трекер с limit=0.0, треканём что-то, чтобы cost > 0
        tracker = CostTracker(limit=0.0)
        tracker.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=0, total_tokens=200)
        with pytest.raises(CostLimitExceeded):
            tracker.check_limit()

    def test_limit_not_exceeded(self):
        tracker = CostTracker(limit=1000.0)
        tracker.track(model_name="deepseek-r1", prompt_tokens=1000000, cached_tokens=0, total_tokens=2000000)
        assert tracker.total_cost_rub < 1000.0

    def test_reset_clears_limit_state(self):
        tracker = CostTracker(limit=0.0)
        tracker.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=0, total_tokens=200)
        with pytest.raises(CostLimitExceeded):
            tracker.check_limit()
        tracker.reset()
        # После сброса cost=0, limit=0.0, check_limit: 0.0 >= 0.0 => True
        # Но total_cost=0.0 => check_limit бросит CostLimitExceeded
        with pytest.raises(CostLimitExceeded):
            tracker.check_limit()


class TestCustomPricing:
    """Тесты кастомного прайсинга."""

    def test_custom_pricing_dict(self):
        """Кастомный прайс передаётся как dict, а не JSON-строка."""
        custom_pricing = {
            "custom-model": {
                "input": 2.0,
                "cached": 1.0,
                "output": 3.0,
            }
        }
        tracker = CostTracker(pricing=custom_pricing)
        tracker.track(model_name="custom-model", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        # output = 300-100-50 = 150
        # cost = (100-50)*2.0/1M + 50*1.0/1M + 150*3.0/1M = 100/1M + 50/1M + 450/1M = 600/1M
        expected = (50 * 2.0 + 50 * 1.0 + 150 * 3.0) / 1_000_000
        assert tracker.total_cost_rub == pytest.approx(expected, rel=1e-9)

    def test_custom_pricing_overrides_model(self):
        """Кастомный прайс имеет приоритет над дефолтным для модели."""
        # default pricing для deepseek-r1: input=0.14, cached=0.07, output=0.55
        t1 = CostTracker()
        t1.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=50, total_tokens=300)

        # custom pricing: input=10.0, cached=5.0, output=10.0 (дороже)
        custom_pricing = {
            "deepseek-r1": {
                "input": 10.0,
                "cached": 5.0,
                "output": 10.0,
            }
        }
        t2 = CostTracker(pricing=custom_pricing)
        t2.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=50, total_tokens=300)

        assert t2.total_cost_rub > t1.total_cost_rub


class TestEnvironmentVariables:
    """Тесты чтения из env."""

    def test_env_cost_limit(self):
        with patch.dict("os.environ", {"HARNESS_COST_LIMIT": "5.00"}):
            tracker = CostTracker()
            assert tracker.limit == 5.0

    def test_env_model_pricing(self):
        pricing_json = json.dumps({"custom-env-model": {"input": 1.5, "cached": 0.75, "output": 2.0}})
        with patch.dict("os.environ", {"HARNESS_MODEL_PRICING": pricing_json}):
            tracker = CostTracker()
            # Проверим, что кастомная модель доступна через _pricing_map
            assert "custom-env-model" in tracker._pricing_map

    def test_env_cost_limit_not_set(self):
        """Если HARNESS_COST_LIMIT не установлен, limit=None."""
        import os

        env = dict(os.environ)
        env.pop("HARNESS_COST_LIMIT", None)
        with patch.dict("os.environ", env, clear=True):
            tracker = CostTracker()
            assert tracker.limit is None

    def test_env_cost_limit_invalid(self):
        """Невалидный HARNESS_COST_LIMIT игнорируется."""
        with patch.dict("os.environ", {"HARNESS_COST_LIMIT": "not-a-number"}):
            tracker = CostTracker()
            assert tracker.limit is None


class TestThreadSafety:
    """Тесты потокобезопасности."""

    def test_concurrent_tracking(self):
        import threading

        tracker = CostTracker(limit=1000.0)
        errors = []

        def track_batch(batch_id):
            try:
                for _ in range(100):
                    tracker.track(
                        model_name="deepseek-r1",
                        prompt_tokens=10,
                        cached_tokens=5,
                        total_tokens=50,
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=track_batch, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert tracker.total_tokens > 0
        assert tracker.total_cost_rub > 0.0


class TestLogging:
    """Тесты логирования."""

    def test_cost_tracking_result(self):
        """Проверка, что track() возвращает словарь с метриками."""
        tracker = CostTracker()
        result = tracker.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        assert "model" in result
        assert "cost_rub" in result
        assert "total_cost_rub" in result

    def test_cost_tracking_result_fields(self):
        """Проверка полей в result track()."""
        tracker = CostTracker()
        result = tracker.track(model_name="deepseek-r1", prompt_tokens=100, cached_tokens=50, total_tokens=300)
        assert result["model"] == "deepseek-r1"
        assert result["prompt_tokens"] == 100
        assert result["cached_tokens"] == 50
        assert result["total_tokens"] == 300
        assert result["output_tokens"] == 150  # 300 - 100 - 50
        assert result["cost_rub"] > 0.0


class TestDefaultPricing:
    """Тесты DEFAULT_PRICING."""

    def test_deepseek_r1_pricing(self):
        assert DEFAULT_PRICING["deepseek-r1"]["input"] == 0.14
        assert DEFAULT_PRICING["deepseek-r1"]["cached"] == 0.07
        assert DEFAULT_PRICING["deepseek-r1"]["output"] == 0.55

    def test_deepseek_v3_pricing(self):
        assert DEFAULT_PRICING["deepseek-v3"]["input"] == 0.27
        assert DEFAULT_PRICING["deepseek-v3"]["cached"] == 0.07
        assert DEFAULT_PRICING["deepseek-v3"]["output"] == 1.10

    def test_gpt4o_pricing(self):
        assert DEFAULT_PRICING["gpt-4o"]["input"] == 2.50
        assert DEFAULT_PRICING["gpt-4o"]["cached"] == 1.25
        assert DEFAULT_PRICING["gpt-4o"]["output"] == 10.00

    def test_fallback_in_code(self):
        """Fallback pricing определяется в коде track(), а не в DEFAULT_PRICING."""
        tracker = CostTracker()
        # Модели, которой нет в DEFAULT_PRICING, получит fallback 1.0/0.5/1.0
        tracker.track(model_name="brand-new-model", prompt_tokens=1000, cached_tokens=0, total_tokens=2000)
        expected = (1000 * 1.0 + 0 * 0.5 + 1000 * 1.0) / 1_000_000
        assert tracker.total_cost_rub == pytest.approx(expected, rel=1e-9)
