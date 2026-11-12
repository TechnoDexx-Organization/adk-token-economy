import logging

logger = logging.getLogger(__name__)


def build_retry_config(
    max_retries: int = 3,
    timeout_seconds: int = 60,
    backoff_factor: float = 2.0,
) -> dict:
    """
    Возвращает конфигурацию для LiteLlm с настройками повторных попыток.
    
    Args:
        max_retries: Сколько раз пытаться повторить запрос при ошибке.
        timeout_seconds: Таймаут на один запрос (в секундах).
        backoff_factor: Множитель для экспоненциальной задержки между попытками.
    
    Returns:
        Словарь с параметрами для LiteLlm.
    """
    return {
        "max_retries": max_retries,
        "timeout": timeout_seconds,
        "retry_policy": {
            "retry_on": ["429", "500", "502", "503", "504"],  # Коды, при которых повторяем
            "backoff_factor": backoff_factor,
            "max_delay": 10.0,  # Максимальная задержка между попытками
        },
    }


def log_retry_attempt(attempt: int, error: Exception) -> None:
    """Логирует попытку повторного запроса."""
    logger.warning(
        f"⚠️ Retry попытка #{attempt} из-за ошибки: {type(error).__name__}: {error}"
    )