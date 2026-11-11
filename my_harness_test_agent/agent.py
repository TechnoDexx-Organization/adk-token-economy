import os
import pathlib
import subprocess

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# Импортируем из нашей универсальной библиотеки agent-harness
from harness import (
    ColdStorage,
    EmotionAnalyzer,
    ShellCommandGuard,
    SlidingWindow,
    TextSparsifier,
)
from harness.adapters.adk import ADKAgentCallbacks
from harness.modules.skills import build_safe_skill_toolset

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
        "You are an AI assistant running in a token-expensive environment. "
        "You have access to specialized skills (tools) and a shell command executor. "
        "If the user asks to check the environment, install packages, or run scripts, "
        "USE your skills to get the job done. You can read skill instructions, "
        "read skill resources, and execute scripts using the `run_shell_command` tool. "
        "NEVER use 'find /' or search the whole filesystem. Only run commands inside the current project. "
        "Otherwise, answer the user's questions directly. "
        "Keep your thoughts short. Minimize the number of tokens in your responses."
    )

# ==========================================
# СБОРКА АГЕНТА ИЗ КОМПОНЕНТОВ
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === ИНСТРУМЕНТ ДЛЯ ВЫПОЛНЕНИЯ КОМАНД ===
def run_shell_command(command: str) -> str:
    """Выполняет shell-команду.
    
    Args:
        command: Команда для выполнения (например, 'ls -la' или 'bash script.sh').
    """
    try:
        # Выполняем команду с таймаутом и рабочей директорией BASE_DIR
        result = subprocess.run(
            command, 
            check=False, shell=True, 
            capture_output=True, 
            text=True, 
            timeout=15,
            cwd=BASE_DIR  # <--- ВАЖНОЕ ИСПРАВЛЕНИЕ!
        )
        output = result.stdout.strip()
        if result.stderr:
            output += f"\n[STDERR]: {result.stderr.strip()}"
        return output or "Команда выполнена успешно (нет вывода)."
    except subprocess.TimeoutExpired:
        return "❌ Ошибка: Команда превысила лимит времени (15 секунд)."
    except Exception as e:  # noqa: BLE001
        return f"❌ Ошибка выполнения: {e!s}"


# 1. Инициализируем модули
analyzer = EmotionAnalyzer()
sparsifier = TextSparsifier()
storage = ColdStorage(BASE_DIR)
window = SlidingWindow(
    max_messages=10,
    max_tokens=12_000,
    max_message_tokens=1_500,
)
shell_guard = ShellCommandGuard()

# 2. Собираем адаптер для ADK
callbacks = ADKAgentCallbacks(
    analyzer=analyzer,
    sparsifier=sparsifier,
    storage=storage,
    window=window,
    tone_hints=TONE_HINTS,
    guards=[shell_guard],
)

# 3. Загружаем Skills (Навыки)
skills_dir = pathlib.Path(BASE_DIR) / "skills"
tools_list = [run_shell_command]

if skills_dir.exists():
    print(f"📦 Загрузка навыков из {skills_dir}...")
    try:
        my_skill_toolset = build_safe_skill_toolset(skills_dir)
        tools_list.append(my_skill_toolset)
        print("✅ Навыки успешно загружены и проверены!")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Ошибка при загрузке навыков: {e}")

# 4. Создаем агента
root_agent = Agent(
    model=get_llm_model(),
    name="harness_test_agent",
    description="Test agent with universal harness and skills.",
    instruction=get_static_instruction,
    tools=tools_list,
    before_model_callback=callbacks.before_model,
    after_model_callback=callbacks.after_model,
    before_tool_callback=callbacks.before_tool,
)