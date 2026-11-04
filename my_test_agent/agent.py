import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Импортируем из нашей универсальной библиотеки agent-harness
from harness import (
    ColdStorage,
    EmotionAnalyzer,
    #SafeToolGuard,
    SlidingWindow,
    TextSparsifier,
)
from harness.adapters.adk import ADKAgentCallbacks

load_dotenv()

# === НАСТРОЙКА ТОНА ===
TONE_HINTS = {
    "sadness": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "fear": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "anger": "The user is irritated or angry. Be calm, objective, and concise.",
    "disgust": "The user is irritated or angry. Be calm, objective, and concise.",
    "happiness": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
    "enthusiasm": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
}


def get_llm_model():
    """Возвращает настроенный LiteLlm в зависимости от AI_PROVIDER."""
    provider = os.getenv("AI_PROVIDER", "yandex").lower()

    if provider == "selectel":
        print("🔌 Используем Selectel AI Gateway")
        return LiteLlm(
            model="openai/deepseek/deepseek-v4.1-flash",
            api_base="https://api.selectel.ru/aig/v1",
            api_key=os.getenv("SELECTEL_API_KEY"),
            reasoning_effort="off",
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
        )
        
    raise ValueError(f"❌ Неизвестный AI_PROVIDER: {provider}")


def get_static_instruction(context) -> str:
    """Статичная инструкция для кэширования."""
    return (
        "You are an echo agent. Forward the user's messages. "
        "IMPORTANT: You are running in a token-expensive environment. "
        "DO NOT call any tools. Just output the user's exact text. "
        "Minimize the number of tokens in your responses. "
        "Keep your thoughts short. Do not over-explain. "
    )


# ==========================================
# СБОРКА АГЕНТА ИЗ КОМПОНЕНТОВ
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Инициализируем модули
analyzer = EmotionAnalyzer()
sparsifier = TextSparsifier()
storage = ColdStorage(BASE_DIR)
window = SlidingWindow(max_events=10)

# 2. Собираем адаптер для ADK
callbacks = ADKAgentCallbacks(
    analyzer=analyzer,
    sparsifier=sparsifier,
    storage=storage,
    window=window,
    tone_hints=TONE_HINTS,
)

# 3. Создаем агента
root_agent = Agent(
    model=get_llm_model(),
    name="echo_agent",
    description="Echo agent with universal harness.",
    instruction=get_static_instruction,
    tools=[],
    before_model_callback=callbacks.before_model,
    after_model_callback=callbacks.after_model,
)