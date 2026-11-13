import os
import pathlib

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import ToolContext

# Импортируем из нашей универсальной библиотеки agent-harness
from harness import (
    ColdStorage,
    EmotionAnalyzer,
    FactMemoryProvider,
    FileMemoryProvider,
    FileTools,
    HttpMemoryProvider,
    ShellCommandGuard,
    SlidingWindow,
    TextSparsifier,
)
from harness.adapters.adk import ADKAgentCallbacks
from harness.modules.retry import build_retry_config
from harness.modules.secret_editor import SecretEditor
from harness.modules.skills import build_safe_skill_toolset

load_dotenv()


def get_bool_env(key: str, default: bool = True) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


TONE_HINTS = {
    "sadness": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "fear": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "anger": "The user is irritated or angry. Be calm, objective, and concise.",
    "disgust": "The user is irritated or angry. Be calm, objective, and concise.",
    "happiness": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
    "enthusiasm": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
}


def get_llm_model():
    provider = os.getenv("AI_PROVIDER", "yandex").lower()

    retry_kwargs = {}
    if get_bool_env("ENABLE_RETRY", True):
        retry_kwargs = build_retry_config(
            max_retries=int(os.getenv("RETRY_MAX_ATTEMPTS", "3")),
            timeout_seconds=int(os.getenv("RETRY_TIMEOUT_SECONDS", "60")),
            backoff_factor=float(os.getenv("RETRY_BACKOFF_FACTOR", "2.0")),
        )
        print(f"🔁 Retry Logic включен: max_retries={retry_kwargs.get('max_retries')}, timeout={retry_kwargs.get('timeout')}s")
    else:
        print("🚫 Retry Logic отключен (ENABLE_RETRY=false)")

    if provider == "selectel":
        print("🔌 Используем Selectel AI Gateway")
        return LiteLlm(
            model="openai/deepseek/deepseek-v4.1-flash",
            api_base="https://api.selectel.ru/aig/v1",
            api_key=os.getenv("SELECTEL_API_KEY"),
            reasoning_effort="off",
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


def get_memory_provider():
    if not get_bool_env("ENABLE_MEMORY", True):
        print("🚫 Память (RAG) отключена.")
        return None

    provider_type = os.getenv("MEMORY_PROVIDER_TYPE", "file").lower()

    if provider_type == "http" and (url := os.getenv("MEMORY_HTTP_URL")):
        print(f"🧠 Используем внешний RAG: {url}")
        return HttpMemoryProvider(url, api_key=os.getenv("MEMORY_HTTP_API_KEY", ""))

    print("🧠 Используем локальный файловый поиск (cold_storage.jsonl)")
    return FileMemoryProvider(os.path.join(BASE_DIR, "cold_storage.jsonl"))


def get_static_instruction(context) -> str:
    if get_bool_env("ENABLE_PROMPT_FILE", True):
        prompt_path = pathlib.Path(BASE_DIR) / "prompts" / "agent_instruction.md"
        if prompt_path.exists():
            try:
                print(f"📄 Загружена инструкция из файла: {prompt_path}")
                return prompt_path.read_text(encoding="utf-8")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Ошибка чтения файла промпта: {e}. Используем дефолтную инструкцию.")
        else:
            print(f"⚠️ Файл промпта не найден: {prompt_path}. Используем дефолтную инструкцию.")

    hitl_instruction = ""
    if get_bool_env("ENABLE_HITL", True):
        hitl_instruction = (
            "CRITICAL: Before executing `run_shell_command` or `write_file`, you MUST ask the user for permission. "
            "When you try to call one of these tools, the system will block it and return a 'confirmation_required' message with an `args_hash`. "
            "You must then output a message asking the user: 'Can I execute this command/write this file?'. "
            "If the user says 'yes' or 'разрешаю', you must call `approve_action` with the tool name and hash, and then call the original tool again. "
        )

    return (
        "You are an autonomous AI coding assistant running in a token-expensive environment. "
        "You have access to specialized skills (tools) and a shell command executor. "
        "You can read files using `read_file`, write code to files using `write_file`, "
        "and execute them using `run_shell_command` (e.g., `python script.py`). "
        f"{hitl_instruction}"
        "When given a task, write the code, run it to test it, and if there are errors, read the file, fix the errors, and try again. "
        "CRITICAL: NEVER write to system files, hidden config files, or outside the current project directory. "
        "CRITICAL: NEVER use 'find /' or search the whole filesystem. "
        "Keep your thoughts short. Minimize the number of tokens in your responses."
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

analyzer = EmotionAnalyzer() if get_bool_env("ENABLE_EMOTION", True) else None
sparsifier = TextSparsifier() if get_bool_env("ENABLE_SPARSIFIER", True) else None
storage = ColdStorage(BASE_DIR) if get_bool_env("ENABLE_STORAGE", True) else None

if get_bool_env("ENABLE_WINDOW", True):
    window = SlidingWindow(
        max_messages=10,
        max_tokens=12_000,
        max_message_tokens=1_500,
    )
else:
    window = None

shell_guard = ShellCommandGuard() if get_bool_env("ENABLE_GUARD", True) else None
guards_list = [shell_guard] if shell_guard else []

file_tools = FileTools(BASE_DIR)
memory_provider = get_memory_provider()

# Память фактов (всегда включена, если есть ENABLE_FACTS)
if get_bool_env("ENABLE_FACTS", True):
    fact_memory = FactMemoryProvider(os.path.join(BASE_DIR, "facts.jsonl"))
    print("🧠 Память фактов (facts.jsonl) включена.")
else:
    fact_memory = None

if get_bool_env("ENABLE_SECRET_EDITOR", True):
    allowlist_path = os.getenv("SECRET_ALLOWLIST_PATH")
    if allowlist_path and not os.path.isabs(allowlist_path):
        allowlist_path = os.path.join(BASE_DIR, allowlist_path)
    secret_editor = SecretEditor(allowlist_path=allowlist_path)
else:
    secret_editor = None

callbacks = ADKAgentCallbacks(
    analyzer=analyzer,
    sparsifier=sparsifier,
    storage=storage,
    window=window,
    tone_hints=TONE_HINTS,
    guards=guards_list,
    secret_editor=secret_editor,
    enable_hitl=get_bool_env("ENABLE_HITL", True),
    memory_provider=memory_provider,
    fact_memory_provider=fact_memory,  # <--- НОВОЕ
)


def approve_action(tool_name: str, args_hash: str, tool_context: ToolContext) -> str:
    state_key = f"hitl_approved_{tool_name}_{args_hash}"
    tool_context.state[state_key] = True
    return f"✅ Действие '{tool_name}' с хэшем {args_hash} одобрено пользователем. Вы можете снова вызвать этот инструмент."


def remember_fact(fact: str, tool_context: ToolContext) -> str:
    """Запоминает важный факт о пользователе.
    
    Args:
        fact: Факт, который нужно запомнить (например, "Игорь не любит звонки").
    """
    if fact_memory is None:
        return "❌ Память фактов отключена."
    fact_memory.add_fact(fact, source="agent")
    return f"✅ Факт запомнен: {fact}"


skills_dir = pathlib.Path(BASE_DIR) / "skills"
tools_list = [
    file_tools.run_shell_command,
    file_tools.write_file,
    file_tools.read_file,
]

if get_bool_env("ENABLE_HITL", True):
    tools_list.append(approve_action)

if fact_memory is not None:
    tools_list.append(remember_fact)

if skills_dir.exists():
    print(f"📦 Загрузка навыков из {skills_dir}...")
    try:
        my_skill_toolset = build_safe_skill_toolset(skills_dir)
        tools_list.append(my_skill_toolset)
        print("✅ Навыки успешно загружены и проверены!")
    except Exception as e:  # noqa: BLE001
        print(f"❌ Ошибка при загрузке навыков: {e}")

root_agent = Agent(
    model=get_llm_model(),
    name="harness_test_agent",
    description="Autonomous AI coding agent with universal harness, skills, feature flags, RAG and facts memory.",
    instruction=get_static_instruction,
    tools=tools_list,
    before_model_callback=callbacks.before_model,
    after_model_callback=callbacks.after_model,
    before_tool_callback=callbacks.before_tool,
)