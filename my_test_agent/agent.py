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

# ==========================================
# 1. КЛАСС ДЛЯ АНАЛИЗА ЭМОЦИЙ
# ==========================================
class EmotionAnalyzer:
    def __init__(self):
        print("⏳ Загрузка модели эмоционального анализа...")
        try:
            self.pipeline = pipeline(
                "text-classification",
                model="Aniemore/rubert-tiny2-russian-emotion-detection",
            )
            print("✅ Модель эмоционального анализа загружена")
        except Exception as e:  # noqa: BLE001
            self.pipeline = None
            print(f"⚠️ Не удалось загрузить модель: {e}")

    def analyze(self, text: str) -> str:
        if not self.pipeline:
            return "neutral"
        try:
            result = self.pipeline(text, truncation=True, max_length=512)[0]
            print(f"  (Эмоция: {result['label']} {result['score']:.2f})")
            return result["label"]
        except (TypeError, KeyError, IndexError) as e:
            print(f"  (Не удалось определить эмоцию: {e})")
            return "neutral"


# ==========================================
# 2. КЛАСС ДЛЯ СУШКИ ТЕКСТА
# ==========================================
class TextSparsifier:
    def __init__(self):
        self.filler_words = [
            r"\bкороче\b", r"\bв общем\b", r"\bкак бы\b", r"\bтипа\b",
            r"\bна самом деле\b", r"\bв принципе\b", r"\bпо сути\b",
            r"\bсобственно\b", r"\bпросто\b", r"\bзначит\b",
        ]

    def sparsify(self, text: str) -> str:
        """Удаляет воду и связки."""
        for word in self.filler_words:
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


# ==========================================
# 3. КЛАСС ДЛЯ ЛЕДНИКА (COLD STORAGE)
# ==========================================
class ColdStorage:
    def __init__(self, base_dir: str):
        self.file_path = os.path.join(base_dir, "cold_storage.jsonl")

    def log(self, session_id: str, role: str, text: str):
        entry = {
            "session_id": session_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "text": text,
        }
        try:
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Ошибка записи в ледник: {e}")


# ==========================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def event_to_text(event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text)


def get_static_instruction(context) -> str:
    return (
        "You are an echo agent. Forward the user's messages. "
        "IMPORTANT: You are running in a token-expensive environment. "
        "DO NOT call any tools. Just output the user's exact text. "
        "Minimize the number of tokens in your responses. "
        "Keep your thoughts short. Do not over-explain. "
    )


def get_llm_model():
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


# ==========================================
# 5. КЛАСС ДЛЯ КОЛБЭКОВ АГЕНТА (МОЗГ)
# ==========================================
class AgentCallbacks:
    def __init__(self, analyzer: EmotionAnalyzer, sparsifier: TextSparsifier, storage: ColdStorage):
        self.analyzer = analyzer
        self.sparsifier = sparsifier
        self.storage = storage
        self.flag_key = "_processed"
        self.window_size = 10
        self.tone_hints = {
            "sadness": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
            "fear": "The user is feeling sad or fearful. Be gentle, supportive, and brief.",
            "anger": "The user is irritated or angry. Be calm, objective, and concise.",
            "disgust": "The user is irritated or angry. Be calm, objective, and concise.",
            "happiness": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
            "enthusiasm": "The user is in a good mood. Be slightly more upbeat, but remain efficient.",
        }

    def before_model(self, callback_context: CallbackContext, llm_request: LlmRequest):
        if callback_context.state.get(self.flag_key):
            return
        session = callback_context.session
        if not session:
            return

        # 1. ЛЕДНИК: Оригинал последнего сообщения
        for event in reversed(session.events):
            if event.author == "user":
                if orig_text := event_to_text(event):
                    self.storage.log(session.id, "user", orig_text)
                break

        # 2. ГОРЯЧИЙ КОНТЕКСТ: Окно и сушка
        all_valid_events = []
        for event in session.events:
            text = event_to_text(event)
            if not text:
                continue
            role = "user" if event.author == "user" else "model"
            if role == "user":
                callback_context.state["current_emotion"] = self.analyzer.analyze(text)
                text = self.sparsifier.sparsify(text)
            all_valid_events.append((role, text))

        windowed_events = all_valid_events[-self.window_size:]
        new_contents = [Content(role=role, parts=[Part(text=text)]) for role, text in windowed_events]

        # 3. ДИНАМИЧЕСКИЙ ТОН
        emotion = callback_context.state.get("current_emotion", "neutral")
        tone_hint = self.tone_hints.get(emotion, "Neutral. Be direct and helpful.")

        if new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
            part = new_contents[-1].parts[0]
            prefix = f"{part.text}\n\n" if part.text else ""
            part.text = f"{prefix}[Tone hint: {tone_hint}]"

        print(f"=== ИСТОРИЯ (событий в окне: {len(new_contents)} из {len(all_valid_events)}) ===")
        llm_request.contents = new_contents
        callback_context.state[self.flag_key] = True

    def after_model(self, callback_context: CallbackContext, llm_response: LlmResponse):
        is_empty = (
            not llm_response.content
            or not llm_response.content.parts
            or not any(p.text for p in llm_response.content.parts)
        )

        if is_empty:
            self._handle_empty_response(callback_context, llm_response)
            return

        has_function_call = any(part.function_call is not None for part in llm_response.content.parts)

        if reply_text := "".join(part.text for part in llm_response.content.parts if part.text):
            print("=== ОТВЕТ МОДЕЛИ ===")
            print(reply_text)
            print("=" * 20)

        if llm_response.usage_metadata:
            cached = getattr(llm_response.usage_metadata, "cached_content_token_count", 0)
            prompt = getattr(llm_response.usage_metadata, "prompt_token_count", 0)
            print(f"📊 Метрики: prompt={prompt}, cached={cached}")

        if not has_function_call:
            callback_context.state[self.flag_key] = False
            print("=== [after_model] финал, флаг снят ===")

    def _handle_empty_response(self, callback_context: CallbackContext, llm_response: LlmResponse):
        callback_context.state[self.flag_key] = False
        error_count = callback_context.state.get("error_count", 0) + 1
        callback_context.state["error_count"] = error_count

        reason = f"Сбой API: {llm_response.error_code}" if llm_response.error_code else "Модель вернула пустоту"
        print(f"⚠️ [after_model] {reason}. Сбой №{error_count}.")

        if session := callback_context.session:
            for event in reversed(session.events):
                if event.author == "user":
                    if text := event_to_text(event):
                        llm_response.content = Content(role="model", parts=[Part(text=text)])
                        llm_response.error_code = None
                        llm_response.error_message = None
                    break


# ==========================================
# 6. СБОРКА АГЕНТА
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Инициализируем компоненты
analyzer = EmotionAnalyzer()
sparsifier = TextSparsifier()
storage = ColdStorage(BASE_DIR)

# Создаем "мозг" (колбэки) и передаем ему компоненты
callbacks = AgentCallbacks(analyzer, sparsifier, storage)

# Создаем самого агента
root_agent = Agent(
    model=get_llm_model(),
    name="echo_agent",
    description="Echo agent.",
    instruction=get_static_instruction,
    tools=[],
    before_model_callback=callbacks.before_model,
    after_model_callback=callbacks.after_model,
)