from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part


class AgentCallbacks:
    def __init__(
        self,
        analyzer=None,
        sparsifier=None,
        storage=None,
        window=None,
        tone_hints=None,
    ):
        self.analyzer = analyzer
        self.sparsifier = sparsifier
        self.storage = storage
        self.window = window
        self.tone_hints = tone_hints or {}
        self.flag_key = "_processed"

    def _event_to_text(self, event) -> str:
        if not event.content or not event.content.parts:
            return ""
        return "".join(part.text for part in event.content.parts if part.text)

    def before_model(self, callback_context: CallbackContext, llm_request: LlmRequest):
        # sourcery skip: low-code-quality
        if callback_context.state.get(self.flag_key):
            return
        session = callback_context.session
        if not session:
            return

        # 1. ЛЕДНИК (опционально)
        if self.storage:
            for event in reversed(session.events):
                if event.author == "user":
                    if orig_text := self._event_to_text(event):
                        self.storage.log(session.id, "user", orig_text)
                    break

        # 2. ПРОХОД ПО ИСТОРИИ (Сушка и Эмоции)
        all_valid_events = []
        for event in session.events:
            text = self._event_to_text(event)
            if not text:
                continue
            role = "user" if event.author == "user" else "model"

            if role == "user":
                # Анализ эмоций (опционально)
                if self.analyzer:
                    emotion = self.analyzer.analyze(text)
                    callback_context.state["current_emotion"] = emotion
                
                # Сушка текста (опционально)
                if self.sparsifier:
                    text = self.sparsifier.apply(text)

            all_valid_events.append((role, text))

        # 3. СКОЛЬЗЯЩЕЕ ОКНО (опционально)
        if self.window:
            windowed_events = self.window.apply(all_valid_events)
        else:
            windowed_events = all_valid_events

        # Собираем Content
        new_contents = [
            Content(role=role, parts=[Part(text=text)]) 
            for role, text in windowed_events
        ]

        # 4. ДИНАМИЧЕСКИЙ ТОН (опционально)
        if self.analyzer and self.tone_hints:
            emotion = callback_context.state.get("current_emotion", "neutral")
            tone_hint = self.tone_hints.get(emotion)
            if tone_hint and new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
                part = new_contents[-1].parts[0]
                prefix = f"{part.text}\n\n" if part.text else ""
                part.text = f"{prefix}[Tone hint: {tone_hint}]"

        print(f"=== ИСТОРИЯ (событий: {len(new_contents)}) ===")
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

        has_function_call = any(
            part.function_call is not None for part in llm_response.content.parts
        )

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

        reason = (
            f"Сбой API: {llm_response.error_code}" 
            if llm_response.error_code 
            else "Модель вернула пустоту"
        )
        print(f"⚠️ [after_model] {reason}. Сбой №{error_count}.")

        if session := callback_context.session:
            for event in reversed(session.events):
                if event.author == "user":
                    if text := self._event_to_text(event):
                        llm_response.content = Content(role="model", parts=[Part(text=text)])
                        llm_response.error_code = None
                        llm_response.error_message = None
                    break
