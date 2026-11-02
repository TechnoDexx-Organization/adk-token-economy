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
# Загружаем пайплайн один раз при старте агента.
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
# =========================================

STRICT_MODE = False
FLAG_KEY = "_processed"

# === НАСТРОЙКА СКОЛЬЗЯЩЕГО ОКНА ===
# Сколько последних реплик (событий) отправлять в модель.
# Всё, что было раньше, отсекается, чтобы не тащить всю историю.
HISTORY_WINDOW_SIZE = 10


def get_dynamic_instruction(context) -> str:
    """
    Динамическая инструкция: читает эмоцию из state и формирует системный промпт.
    Это позволяет модели отвечать с нужным тоном, даже если текст был "высушен".
    """
    # Достаём эмоцию, которую сохранил before_model_callback
    emotion = context.state.get("current_emotion", "neutral")
    
    # Базовые правила (экономия токенов + жесткий запрет на инструмент)
    base_rules = (
        "You are an echo agent. Forward the user's messages. "
        "IMPORTANT: You are running in a token-expensive environment. "
        "DO NOT call the get_echo tool. Just output the user's exact text. "
        "Minimize the number of tokens in your responses. "
        "Keep your thoughts short. Do not over-explain. "
    )
    
    # Добавляем эмоциональный окрас
    if emotion in ["sadness", "fear"]:
        return base_rules + " TONE: The user is feeling sad or fearful. Be gentle, supportive, and brief. Acknowledge their state softly."
    elif emotion in ["anger", "disgust"]:
        return base_rules + " TONE: The user is irritated or angry. Be calm, objective, and concise. Do not mirror their aggression."
    elif emotion in ["happiness", "enthusiasm"]:
        return base_rules + " TONE: The user is in a good mood. You can be slightly more upbeat, but remain efficient."
    
    # Нейтральный или неизвестный
    return base_rules + " TONE: Neutral. Be direct and helpful."


def event_to_text(event) -> str:
    """Достаёт полный текст из события. Пустая строка, если текста нет."""
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text)


def sparsify_text(text: str) -> str:
    """
    "Сушка" текста: убираем воду, делаем предложения короче.
    Экономит около 25% токенов при отправке в модель.
    """
    # 1. Убираем частые «вводные» слова и связки
    filler_words = [
        r"\bкороче\b", r"\bв общем\b", r"\bкак бы\b", r"\bтипа\b",
        r"\bна самом деле\b", r"\bв принципе\b", r"\bпо сути\b",
        r"\bсобственно\b", r"\bпросто\b", r"\bзначит\b"
    ]
    for word in filler_words:
        text = re.sub(word, "", text, flags=re.IGNORECASE)

    # 2. Заменяем длинные связки на точки (разбиваем предложения)
    text = text.replace(" потому что ", ". ")
    text = text.replace(" так как ", ". ")
    text = text.replace(" и теперь ", ". ")
    text = text.replace(", и ", ". ")
    text = text.replace(", а ", ". ")

    # 3. Убираем артефакты сушки (запятые перед точками)
    text = text.replace(",.", ".")
    text = text.replace(", .", ".")

    # 4. Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text).strip()
    
    # 5. Делаем первую букву заглавной
    if text:
        text = text[0].upper() + text[1:]

    return text


def prepare_text(callback_context: CallbackContext, text: str, role: str = "user") -> str:
    """
    Подготавливает текст перед отправкой в модель.
    1. Анализирует эмоцию и сохраняет её в state (для динамической инструкции).
    2. "Сушит" текст для экономии токенов.
    """
    if role != "user":
        return text

    # 1. Определяем эмоцию
    emotion_label = "neutral"
    if emotion_pipeline:
        try:
            result = emotion_pipeline(text, truncation=True, max_length=512)[0]
            emotion_label = result['label']
            score = result['score']
            
            # Сохраняем эмоцию в состояние сессии для get_dynamic_instruction
            callback_context.state["current_emotion"] = emotion_label
            print(f"  (Эмоция определена: {emotion_label} {score:.2f})")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Ошибка при анализе эмоции: {e}")
    
    # 2. Сушим текст (эмоцию убираем, так как она уже в state)
    dry_text = sparsify_text(text)
    
    return dry_text


def my_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
):
    """Вызывается ДО запроса к модели.

    `llm_request.contents` — это список объектов `Content`, которые уйдут
    в модель. Каждый `Content` имеет:
      - role: "user" или "model" (не "assistant"!)
      - parts: список `Part`, у каждого может быть text / function_call / function_response

    ПРАВИЛЬНЫЕ СПОСОБЫ МЕНЯТЬ ДАННЫЕ:

    1) Заменить всё содержимое на новое (как здесь):
         llm_request.contents = new_contents
       Важно: сохраняй чередование ролей user → model → user → model.

    2) Изменить только последнее сообщение пользователя (например, сжать):
         if llm_request.contents:
             last = llm_request.contents[-1]
             if last.role == "user" and last.parts:
                 for p in last.parts:
                     if p.text:
                         p.text = compress(p.text)
       Это безопаснее: остальная история не трогается.

    3) Добавить системное сообщение в начало (не заменяя историю):
         llm_request.contents.insert(
             0, Content(role="user", parts=[Part(text="[system]: будь краток")])
         )
       Внимание: роль "system" в ADK часто передаётся отдельно через
       instruction, а не через contents. Используй только если уверен.

    4) Отфильтровать лишние события (например, убрать вызовы инструментов):
         llm_request.contents = [
             c for c in llm_request.contents
             if not any(p.function_call for p in c.parts)
         ]
       Осторожно: модель может потерять контекст и зациклиться.

    ЧЕГО НЕ ДЕЛАТЬ:
      - Не схлопывай всё в одно сообщение (роли сломаются, модель зациклится).
      - Не смешивай роли: "model" — это ответ ассистента, "user" — всё остальное.
      - Не меняй `llm_request.contents` на пустой список — модель упадёт.
    """
    # Защита от повторного наложения
    if callback_context.state.get(FLAG_KEY):
        print("=== [before_model] уже обработано, пропускаем ===")
        return

    session = callback_context.session
    if not session:
        return

    # 1. Собираем ВСЕ валидные события (только текст, без вызовов инструментов)
    all_valid_events = []
    for event in session.events:
        text = event_to_text(event)
        if not text:
            continue
        role = "user" if event.author == "user" else "model"
        all_valid_events.append((role, text))

    # 2. ПРИМЕНЯЕМ СКОЛЬЗЯЩЕЕ ОКНО
    # Берём только последние HISTORY_WINDOW_SIZE реплик.
    windowed_events = all_valid_events[-HISTORY_WINDOW_SIZE:]

    # 3. Подготавливаем текст (сушка + эмоции) только для событий из окна
    new_contents = []
    for role, text in windowed_events:
        processed_text = prepare_text(callback_context, text, role=role)
        new_contents.append(Content(role=role, parts=[Part(text=processed_text)]))

    print(f"=== ИСТОРИЯ (событий в окне: {len(new_contents)} из {len(all_valid_events)}) ===")
    for c in new_contents:
        preview = c.parts[0].text[:80] if c.parts and c.parts[0].text else ""
        print(f"  [{c.role}]: {preview}")
    print("=" * 50)

    # 4. Отправляем в модель
    llm_request.contents = new_contents

    # Поднимаем флаг: в этом ходе больше не обрабатываем
    callback_context.state[FLAG_KEY] = True


def my_after_model_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
):
    """Вызывается ПОСЛЕ ответа модели.

    `llm_response.content` — это `Content` (или None при ошибке).
    У него есть `parts`, и в каждом `Part` может быть:
      - text: строка ответа
      - function_call: вызов инструмента (name, args)
      - function_response: результат инструмента (если это не финал)

    ПРАВИЛЬНЫЕ СПОСОБЫ МЕНЯТЬ ДАННЫЕ:

    1) Изменить текст в ответе (например, добавить тепло):
         for p in llm_response.content.parts:
             if p.text:
                 p.text = add_warmth(p.text)
       `Part` — pydantic-модель, поля мутабельны.

    2) Заменить весь ответ на новый текст (например, переформулировать):
         llm_response.content = Content(
             role="model",
             parts=[Part(text="готовый ответ")],
         )
       Роль оставляй "model", иначе следующий ход может сломаться.

    3) Добавить недостающую часть (например, если модель забыла текст):
         if not any(p.text for p in llm_response.content.parts):
             llm_response.content.parts.append(Part(text="..."))
       Осторожно: если там был function_call, добавление текста может
       сбить агентный цикл.

    4) Перехватить вызов инструмента и подменить аргументы:
         for p in llm_response.content.parts:
             if p.function_call and p.function_call.name == "get_echo":
                 p.function_call.args["string"] = "подменённый аргумент"
       Полезно для отладки, но легко получить неожиданное поведение.

    ЧЕГО НЕ ДЕЛАТЬ:
      - Не пытайся в `after_model` "отменить" вызов инструмента —
        для этого есть `before_tool_callback`.
      - Не меняй роль с "model" на "user" — это ломает историю.
      - Не добавляй `function_response` вручную — его должен
        сформировать сам агент после реального вызова инструмента.
    """
    # Обработка ошибок: если модель упала — снимаем флаг, чтобы не залипнуть
    if llm_response.error_code:
        callback_context.state[FLAG_KEY] = False
        print(f"=== [after_model] ошибка {llm_response.error_code}, флаг снят ===")
        if llm_response.error_message:
            print(f"    Сообщение: {llm_response.error_message}")
        return

    if not llm_response.content or not llm_response.content.parts:
        return

    has_function_call = any(
        part.function_call is not None for part in llm_response.content.parts
    )

    # Логируем ответ (и здесь же можно менять текст)
    for part in llm_response.content.parts:
        if part.text:
            print("=== ОТВЕТ МОДЕЛИ ===")
            print(part.text)
            print("=" * 20)
            # Пример изменения:
            # part.text = "✨ " + part.text

    # Если это финальный ответ (без вызова инструмента) — снимаем флаг,
    # чтобы следующий ход пользователя обработался заново.
    if not has_function_call:
        callback_context.state[FLAG_KEY] = False
        print("=== [after_model] финал, флаг снят ===")


def get_echo(string: str) -> str:
    """Returns the input string as is."""
    return string


root_agent = Agent(
    model=LiteLlm(
        model="openai/gpt://b1g0mvagt30sttlfi76u/deepseek-v4.1-flash/latest",
        api_base=os.getenv("YANDEX_API_BASE"),
        api_key=os.getenv("YANDEX_API_KEY"),
        extra_headers={"OpenAI-Project": os.getenv("YANDEX_FOLDER_ID")},
    ),
    name='echo_agent',
    description='Echo agent.',
    # ИСПОЛЬЗУЕМ ДИНАМИЧЕСКУЮ ИНСТРУКЦИЮ
    instruction=get_dynamic_instruction,
    tools=[get_echo],
    before_model_callback=my_before_model_callback,
    after_model_callback=my_after_model_callback,
)