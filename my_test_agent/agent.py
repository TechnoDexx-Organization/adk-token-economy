import json
import os
import re
from datetime import datetime, timezone

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
        "text-classification", model="Aniemore/rubert-tiny2-russian-emotion-detection"
    )
    print("✅ Модель эмоционального анализа загружена")
except Exception as e:  # noqa: BLE001
    emotion_pipeline = None
    print(f"⚠️ Не удалось загрузить модель: {e}")

STRICT_MODE = False
FLAG_KEY = "_processed"
HISTORY_WINDOW_SIZE = 10

# Одинаковые подсказки схлопнуты в один ключ — значения не дублируются
TONE_HINTS = {
    "sadness": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "fear": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
    "anger": "The user is irritated or angry. Be calm, objective, and concise.",
    "disgust": "The user is irritated or angry. Be calm, objective, and concise.",
    "happiness": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
    "enthusiasm": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
}

# === НАСТРОЙКА ЛЕДНИКА (COLD STORAGE) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLD_STORAGE_FILE = os.path.join(BASE_DIR, "cold_storage.jsonl")


def log_to_cold_storage(session_id: str, role: str, text: str):
    """Пишет ОРИГИНАЛЬНЫЙ текст в ледник."""
    entry = {
        "session_id": session_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "text": text,
    }
    try:
        with open(COLD_STORAGE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"⚠️ Ошибка записи в ледник: {e}")


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
        )
        
    # Sourcery Fix: Убрали else, так как предыдущие if возвращают результат
    raise ValueError(f"❌ Неизвестный AI_PROVIDER: {provider}")


def get_static_instruction(context) -> str:
    """
    СТАТИЧНАЯ инструкция. Она никогда не меняется, поэтому провайдер
    может кэшировать её, экономя наши токены.
    Вся динамика (эмоции) теперь передается через contents.
    """
    return (
        "You are an echo agent. Forward the user's messages. "
        "IMPORTANT: You are running in a token-expensive environment. "
        "DO NOT call any tools. Just output the user's exact text. "
        "Minimize the number of tokens in your responses. "
        "Keep your thoughts short. Do not over-explain. "
    )


def event_to_text(event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text)


def sparsify_text(text: str) -> str:
    """Сушка текста (удаление воды и связок)."""
    filler_words = [
        r"\bкороче\b",
        r"\bв общем\b",
        r"\bкак бы\b",
        r"\bтипа\b",
        r"\bна самом деле\b",
        r"\bв принципе\b",
        r"\bпо сути\b",
        r"\bсобственно\b",
        r"\bпросто\b",
        r"\bзначит\b",
    ]
    for word in filler_words:
        text = re.sub(word, "", text, flags=re.IGNORECASE)
    text = (
        text.replace(" потому что ", ". ")
        .replace(" так как ", ". ")
        .replace(" и теперь ", ". ")
    )
    text = text.replace(", и ", ". ").replace(", а ", ". ")
    text = text.replace(",.", ".").replace(", .", ".")
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def prepare_text(
    callback_context: CallbackContext, text: str, role: str = "user"
) -> str:
    """Анализ эмоций и сушка текста перед отправкой."""
    if role != "user":
        return text

    if emotion_pipeline:
        try:
            result = emotion_pipeline(text, truncation=True, max_length=512)[0]
            callback_context.state["current_emotion"] = result["label"]
            print(f"  (Эмоция: {result['label']} {result['score']:.2f})")
        except (TypeError, KeyError, IndexError) as e:
            print(f"  (Не удалось определить эмоцию: {e})")

    return sparsify_text(text)


def my_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    """Собирает историю, применяет окно, пишет оригинал в ледник и сушит."""
    if callback_context.state.get(FLAG_KEY):
        return
    session = callback_context.session
    if not session:
        return

    # 1. ЛЕДНИК: Сохраняем ОРИГИНАЛ последнего сообщения пользователя до всяких преобразований
    for event in reversed(session.events):
        if event.author == "user":
            # Sourcery Fix: Используем walrus-оператор
            if orig_text := event_to_text(event):
                log_to_cold_storage(session.id, "user", orig_text)
            break

    # 2. ГОРЯЧИЙ КОНТЕКСТ: Собираем и сушим окно
    all_valid_events = [
        ("user" if event.author == "user" else "model", text)
        for event in session.events
        if (text := event_to_text(event))
    ]

    windowed_events = all_valid_events[-HISTORY_WINDOW_SIZE:]
    new_contents = []
    for role, text in windowed_events:
        processed_text = prepare_text(callback_context, text, role=role)
        new_contents.append(Content(role=role, parts=[Part(text=processed_text)]))

    # 3. ДИНАМИЧЕСКИЙ ТОН: Вставляем подсказку в конец последнего сообщения пользователя
    emotion = callback_context.state.get("current_emotion", "neutral")
    tone_hint = TONE_HINTS.get(emotion, "Neutral. Be direct and helpful.")

    # Объединяем if через and (совет линтера)
    if new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
        part = new_contents[-1].parts[0]
        # Если текст есть, добавляем к нему. Если его нет (None), создаем новый.
        if part.text:
            part.text = f"{part.text}\n\n[Tone hint: {tone_hint}]"
        else:
            part.text = f"[Tone hint: {tone_hint}]"

    print(
        f"=== ИСТОРИЯ (событий в окне: {len(new_contents)} из {len(all_valid_events)}) ==="
    )
    llm_request.contents = new_contents
    callback_context.state[FLAG_KEY] = True


def _handle_empty_response(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    """Sourcery Fix: Вынесли логику спасения в отдельную функцию."""
    callback_context.state[FLAG_KEY] = False
    error_count = callback_context.state.get("error_count", 0) + 1
    callback_context.state["error_count"] = error_count
    
    reason = (
        f"Сбой API: {llm_response.error_code}"
        if llm_response.error_code
        else "Модель вернула пустоту"
    )
    print(f"⚠️ [after_model] {reason}. Сбой №{error_count}.")

    if session := callback_context.session:
        for event in reversed(session.events):
            if event.author == "user":
                if text := event_to_text(event):
                    llm_response.content = Content(role="model", parts=[Part(text=text)])
                    llm_response.error_code = None
                    llm_response.error_message = None
                break


def my_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    """Обработка ответа + спасательный круг при пустоте."""
    is_empty = (
        not llm_response.content
        or not llm_response.content.parts
        or not any(p.text for p in llm_response.content.parts)
    )

    if is_empty:
        _handle_empty_response(callback_context, llm_response)
        return

    has_function_call = any(
        part.function_call is not None for part in llm_response.content.parts
    )

    # Логируем ответ (Sourcery Fix: walrus-оператор)
    if reply_text := "".join(part.text for part in llm_response.content.parts if part.text):
        print("=== ОТВЕТ МОДЕЛИ ===")
        print(reply_text)
        print("=" * 20)

    if llm_response.usage_metadata:
        cached = getattr(llm_response.usage_metadata, "cached_content_token_count", 0)
        prompt = getattr(llm_response.usage_metadata, "prompt_token_count", 0)
        print(f"📊 Метрики: prompt={prompt}, cached={cached}")

    if not has_function_call:
        callback_context.state[FLAG_KEY] = False
        print("=== [after_model] финал, флаг снят ===")


root_agent = Agent(
    model=get_llm_model(),
    name="echo_agent",
    description="Echo agent.",
    # ИСПОЛЬЗУЕМ СТАТИЧНУЮ ИНСТРУКЦИЮ
    instruction=get_static_instruction,
    tools=[],
    before_model_callback=my_before_model_callback,
    after_model_callback=my_after_model_callback,
)