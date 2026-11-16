import logging

logger = logging.getLogger(__name__)


def build_retry_config(
    max_retries: int = 3,
    timeout_seconds: int = 60,
    backoff_factor: float = 2.0,
) -> dict:
    """
    Возвращает конфигурацию для LiteLlm с настройками повторных попыток.

    P1-12 (аудит): возвращаем только те ключи, которые LiteLLM понимает.
    - `num_retries` — количество повторов (int)
    - `timeout` — таймаут на запрос (float)
    - `retry_policy` — НЕ поддерживается в `acompletion()`. Раньше он
      проваливался в тело запроса к провайдеру и вызывал 400.

    `backoff_factor` оставлен в сигнатуре для обратной совместимости,
    но LiteLLM использует встроенную экспоненциальную задержку и не
    принимает её как kwarg. При отличии от дефолта — предупреждаем в лог.
    """
    if backoff_factor != 2.0:
        logger.warning(
            "backoff_factor=%s не поддерживается LiteLLM напрямую; "
            "используется встроенная экспоненциальная задержка.",
            backoff_factor,
        )

    return {
        "num_retries": max_retries,
        "timeout": timeout_seconds,
    }


def log_retry_attempt(attempt: int, error: Exception) -> None:
    """Логирует попытку повторного запроса."""
    logger.warning(
        "⚠️ Retry попытка #%s из-за ошибки: %s: %s",
        attempt,
        type(error).__name__,
        error,
    )