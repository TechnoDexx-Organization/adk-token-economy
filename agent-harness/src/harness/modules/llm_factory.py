import os

from google.adk.models.lite_llm import LiteLlm

from .retry import build_retry_config


def _get_bool_env(key: str, default: bool = True) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


def build_llm_model(provider: str | None = None) -> LiteLlm:
    """
    Собирает и возвращает настроенный LiteLlm на основе переменных окружения.
    
    Args:
        provider: Явное указание провайдера (yandex, selectel, selectel_router).
                  Если не указан, берется из AI_PROVIDER (по умолчанию yandex).
    """
    provider = (provider or os.getenv("AI_PROVIDER", "yandex")).lower()

    # Подготавливаем конфигурацию retry (если включена)
    retry_kwargs = {}
    if _get_bool_env("ENABLE_RETRY", True):
        retry_kwargs = build_retry_config(
            max_retries=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
            timeout_seconds=int(os.getenv("RETRY_TIMEOUT_SECONDS", "60")),
            backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        )
        print(f"🔁 Retry Logic включен: max_retries={retry_kwargs.get('max_retries')}, timeout={retry_kwargs.get('timeout')}s")
    else:
        print("🚫 Retry Logic отключен (ENABLE_RETRY=false)")

    if provider == "selectel":
        print("🔌 Используем Selectel AI Gateway (напрямую)")
        return LiteLlm(
            model="openai/deepseek/deepseek-v4.1-flash",
            api_base="https://api.selectel.ru/aig/v1",
            api_key=os.getenv("SELECTEL_API_KEY"),
            reasoning_effort="off",
            **retry_kwargs,
        )

    if provider == "selectel_router":
        router_model = os.getenv("SELECTEL_ROUTER_MODEL", "moonshotai/kimi-k2.6")
        print(f"🔌 Используем Selectel AI Router (модель: {router_model})")
        return LiteLlm(
            model=f"openai/{router_model}",
            api_base="https://api.selectel.ru/aig/v1",
            api_key=os.getenv("SELECTEL_API_KEY"),
            reasoning_effort=os.getenv("SELECTEL_ROUTER_REASONING", "low"),
            **retry_kwargs,
        )

    if provider == "yandex":
        print("🔌 Используем Yandex AI Studio")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        return LiteLlm(
            model=f"openai/gpt://{folder_id}/deepseek-v4.1-flash/latest",
            api_base=os.getenv("YANDEX_API_BASE"),
            api_key=os.getenv("YANDEX_API_KEY"),
            extra_headers={"OpenAI-Project": folder_id},
            reasoning_effort="none",
            **retry_kwargs,
        )

    raise ValueError(f"❌ Неизвестный AI_PROVIDER: {provider}")