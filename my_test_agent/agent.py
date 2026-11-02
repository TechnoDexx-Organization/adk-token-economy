import os
import re

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import LiteLlm
from google.genai.types import Content, Part
from transformers import pipeline

load_dotenv()

# === НАСТРОЙКА ЭМОЦИОНАЛЬНОГО АНАЛИЗА ===
print("⏳ Загрузка модели эмоционального анализа...")
try:
    emotion_pipeline = pipeline(
        "text-classification",
        model="Aniemore/rubert-tiny2-russian-emotion-detection"
    )
    print("✅ Модель эмоционального анализа загружена")
except Exception as e:  # noqa: BLE001
    emotion_pipeline = None
    print(f"⚠️ Не удалось загрузить модель: {e}")

STRICT_MODE = False
FLAG_KEY = "_processed"
HISTORY_WINDOW_SIZE = 10


def get_llm_model():
    """Возвращает настроенный LiteLlm в зависимости от AI_PROVIDER."""
    provider = os.getenv("AI_PROVIDER", "yandex").lower()
    
    if provider == "selectel":
        print("🔌 Используем Selectel AI Gateway")
        return LiteLlm(
            model="openai/deepseek/deepseek-v4.1-flash",
            api_base="https://api.selectel.ru/aig/v1",
            api_key=os.getenv("SELECTEL_API_KEY"),
            # Попытка отключить reasoning, чтобы избежать MODEL_RETURNED_NO_CONTENT
            reasoning_effort='off',
        )
        
    elif provider == "yandex":
        print("🔌 Используем Yandex AI Studio")
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        return LiteLlm(
            model=f"openai/gpt://{folder_id}/deepseek-v4.1-flash/latest",
            api_base=os.getenv("YANDEX_API_BASE"),
            api_key=os.getenv("YANDEX_API_KEY"),
            extra_headers={"OpenAI-Project": folder_id},
        )
    else:
        raise ValueError(f"❌ Неизвестный AI_PROVIDER: {provider}")


def get_dynamic_instruction(context) -> str:
    """
    Динамическая инструкция: читает эмоцию из state и формирует системный промпт.
    """
    emotion = context.state.get("current_emotion", "neutral")
    base_rules = (
        "You are an echo agent. Forward the user's messages. "
        "IMPORTANT: You are running in a token-expensive environment. "
        "DO NOT call any tools. Just output the user's exact text. "
        "Minimize the number of tokens in your responses. "
        "Keep your thoughts short. Do not over-explain. "
    )
    if emotion in ["sadness", "fear"]:
        return base_rules + " TONE: The user is feeling sad or fearful. Be gentle, supportive, and brief."
    elif emotion in ["anger", "disgust"]:
        return base_rules + " TONE: The user is irritated or angry. Be calm, objective, and concise."
    elif emotion in ["happiness", "enthusiasm"]:
        return base_rules + " TONE: The user is in a good mood. Be slightly more upbeat, but remain efficient."
    return base_rules + " TONE: Neutral. Be direct and helpful."


def event_to_text(event) -> str:
    """Достаёт полный текст из события."""
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text)


def sparsify_text(text: str) -> str:
    """Сушка текста (удаление воды и связок)."""
    filler_words = [
        r"\bкороче\b", r"\bв общем\b", r"\bкак бы\b", r"\bтипа\b",
        r"\bна самом деле\b", r"\bв принципе\b", r"\bпо сути\b",
        r"\bсобственно\b", r"\bпросто\b", r"\bзначит\b"
    ]
    for word in filler_words:
        text = re.sub(word, "", text, flags=re.IGNORECASE)
    text = text.replace(" потому что ", ". ").replace(" так как ", ". ").replace(" и теперь ", ". ")
    text = text.replace(", и ", ". ").replace(", а ", ". ")
    text = text.replace(",.", ".").replace(", .", ".")
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def prepare_text(callback_context: CallbackContext, text: str, role: str = "user") -> str:
    """Анализ эмоций и сушка текста перед отправкой."""
    if role != "user":
        return text

    if emotion_pipeline:
        try:
            result = emotion_pipeline(text, truncation=True, max_length=512)[0]
            callback_context.state["current_emotion"] = result['label']
            print(f"  (Эмоция: {result['label']} {result['score']:.2f})")
        except Exception:
            pass
    
    return sparsify_text(text)


def my_before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest):
    """Собирает историю, применяет скользящее окно и сушку."""
    if callback_context.state.get(FLAG_KEY):
        return
    session = callback_context.session
    if not session:
        return

    all_valid_events = []
    for event in session.events:
        text = event_to_text(event)
        if not text:
            continue
        role = "user" if event.author == "user" else "model"
        all_valid_events.append((role, text))

    windowed_events = all_valid_events[-HISTORY_WINDOW_SIZE:]
    new_contents = []
    for role, text in windowed_events:
        processed_text = prepare_text(callback_context, text, role=role)
        new_contents.append(Content(role=role, parts=[Part(text=processed_text)]))

    print(f"=== ИСТОРИЯ (событий в окне: {len(new_contents)} из {len(all_valid_events)}) ===")
    llm_request.contents = new_contents
    callback_context.state[FLAG_KEY] = True


def my_after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse):
    """Логика обработки ответа + спасательный круг при пустоте."""
    
    # ПРОВЕРКА НА ПУСТОТУ (Без хардкода кода ошибки, как советовал коллега)
    is_empty = (
        not llm_response.content 
        or not llm_response.content.parts
        or not any(p.text for p in llm_response.content.parts)
    )

    if is_empty:
        callback_context.state[FLAG_KEY] = False
        
        # Счетчик сбоев
        error_count = callback_context.state.get("error_count", 0) + 1
        callback_context.state["error_count"] = error_count
        
        # Логируем сам факт ошибки (для отладки)
        if llm_response.error_code:
            print(f"⚠️ [after_model] Сбой API: {llm_response.error_code}. Сбой №{error_count}.")
        else:
            print(f"⚠️ [after_model] Модель вернула пустоту. Сбой №{error_count}.")
            
        session = callback_context.session
        if session:
            for event in reversed(session.events):
                if event.author == "user":
                    text = event_to_text(event)
                    if text:
                        # КОЛЛЕГА ПРАВ: Возвращаем ОРИГИНАЛЬНЫЙ текст, без сушки
                        llm_response.content = Content(role="model", parts=[Part(text=text)])
                        llm_response.error_code = None
                        llm_response.error_message = None
                    break
        return

    # Обычная логика логов
    has_function_call = any(
        part.function_call is not None for part in llm_response.content.parts
    )

    for part in llm_response.content.parts:
        if part.text:
            print("=== ОТВЕТ МОДЕЛИ ===")
            print(part.text)
            print("=" * 20)

    if not has_function_call:
        callback_context.state[FLAG_KEY] = False
        print("=== [after_model] финал, флаг снят ===")


root_agent = Agent(
    model=get_llm_model(),
    name='echo_agent',
    description='Echo agent.',
    instruction=get_dynamic_instruction,
    tools=[],
    before_model_callback=my_before_model_callback,
    after_model_callback=my_after_model_callback,
)