import difflib
import hashlib
import json

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part

# Импортируем модули из нашей же библиотеки
from ..modules.analyzer import EmotionAnalyzer
from ..modules.secret_editor import SecretEditor
from ..modules.sparsifier import TextSparsifier
from ..modules.storage import ColdStorage
from ..modules.window import SlidingWindow


class ADKAgentCallbacks:
    """
    Класс-адаптер, который транслирует логику Harness в колбэки Google ADK.
    """

    def __init__(
        self,
        analyzer: EmotionAnalyzer | None = None,
        sparsifier: TextSparsifier | None = None,
        storage: ColdStorage | None = None,
        window: SlidingWindow | None = None,
        tone_hints: dict | None = None,
        guards: list | None = None,
        secret_editor: SecretEditor | None = None,
        enable_hitl: bool = True,
    ):
        self.analyzer = analyzer
        self.sparsifier = sparsifier
        self.storage = storage
        self.window = window
        self.tone_hints = tone_hints or {}
        self.guards = guards or []
        self.secret_editor = secret_editor or SecretEditor()
        self.enable_hitl = enable_hitl
        self.flag_key = "_processed"
        self.ledger_key = "_ledger_upto"

        # Список инструментов, которые требуют подтверждения (Human-in-the-Loop)
        self.hitl_tools = {"run_shell_command", "write_file"}

        # Лимиты для ответов инструментов (чтобы не сломать контекст огромным логом)
        self.tool_response_max_chars = 10_000
        # Храним ID уже обработанных ответов инструментов (защита от повторной обработки)
        self._processed_tool_ids: set[str] = set()

        # Кэши "посчитано один раз на событие"
        self._sparse_cache: dict[str, str] = {}
        self._emotion_cache: dict[str, str] = {}
        self._final_cache: dict[str, str] = {}
        self._session_id: str = ""

    # ─────────────────────────── утилиты ───────────────────────────

    def _event_to_text(self, event) -> str:
        if not event.content or not event.content.parts:
            return ""
        return "".join(part.text for part in event.content.parts if part.text)

    def _event_key(self, event) -> str:
        if event_id := getattr(event, "id", "") or "":
            return event_id
        raw = f"{getattr(event, 'author', '')}|{getattr(event, 'invocation_id', '')}|{self._event_to_text(event)}"
        return "synthetic-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _role_of(self, event) -> str:
        return "user" if getattr(event, "author", "") == "user" else "model"

    @staticmethod
    def _removed_spans(original: str, processed: str) -> list[dict]:
        matcher = difflib.SequenceMatcher(None, original, processed, autojunk=False)
        removed: list[dict] = [
            {"start": i1, "end": i2, "text": original[i1:i2]}
            for tag, i1, i2, _j1, _j2 in matcher.get_opcodes()
            if tag in ("delete", "replace") and i2 > i1
        ]
        return removed

    # ─────────────────────── шаг 1: ледник ───────────────────────

    @staticmethod
    def _has_parts(event) -> bool:
        content = getattr(event, "content", None)
        return bool(content and getattr(content, "parts", None))

    @staticmethod
    def _tool_payload(event) -> str:
        payload = []
        for part in getattr(event.content, "parts", []) or []:
            call = getattr(part, "function_call", None)
            response = getattr(part, "function_response", None)
            if call is not None:
                payload.append({"call": getattr(call, "name", "")})
            elif response is not None:
                payload.append({"response": getattr(response, "name", "")})
        return json.dumps(payload, ensure_ascii=False) if payload else ""

    def _ledger_new_events(self, session, callback_context: CallbackContext) -> None:
        if not self.storage:
            return

        events = list(session.events or [])
        logged_upto = callback_context.state.get(self.ledger_key, 0) or 0
        if logged_upto > len(events):
            logged_upto = 0

        for event in events[logged_upto:]:
            text = self._event_to_text(event)
            kind = "user_message" if self._role_of(event) == "user" else "model_message"
            if not text:
                text = self._tool_payload(event)
                kind = "tool_event"
            if not text:
                continue
            self.storage.log(
                session.id,
                self._role_of(event),
                text,
                invocation_id=getattr(event, "invocation_id", "") or "",
                event_id=self._event_key(event),
                kind=kind,
            )

        callback_context.state[self.ledger_key] = len(events)

    # ──────────── шаг 2: анализ и сушка последнего сообщения ────────────

    def _process_last_user_message(self, session, callback_context: CallbackContext) -> str | None:
        last_user_event = next(
            (
                event
                for event in reversed(session.events or [])
                if getattr(event, "author", "") == "user"
                and self._event_to_text(event)
            ),
            None,
        )
        if last_user_event is None:
            return None

        original = self._event_to_text(last_user_event)
        key = self._event_key(last_user_event)
        invocation_id = getattr(last_user_event, "invocation_id", "") or ""

        if self.analyzer:
            emotion = self._emotion_cache.get(key)
            if emotion is None:
                emotion = self.analyzer.analyze(original)
                self._emotion_cache[key] = emotion
            callback_context.state["current_emotion"] = emotion

        processed = original
        if self.sparsifier:
            cached = self._sparse_cache.get(key)
            if cached is None:
                processed = self.sparsifier.process(original, {})
                self._sparse_cache[key] = processed
                self._final_cache.pop(key, None)
                if self.storage and processed != original:
                    self.storage.log_manifest(
                        session.id,
                        event_id=key,
                        invocation_id=invocation_id,
                        role="user",
                        original=original,
                        processed=processed,
                        removed=self._removed_spans(original, processed),
                        ops=["sparsify"],
                    )
            else:
                processed = cached

        return processed

    # ───────────────────── шаг 3: окно ─────────────────────

    def _final_text(self, event) -> str:
        key = self._event_key(event)
        cached = self._final_cache.get(key)
        if cached is not None:
            return cached

        text = self._event_to_text(event)
        if self._role_of(event) == "user":
            text = self._sparse_cache.get(key, text)

        window = self.window
        if window is not None:
            compacted = window.compact(text)
            if compacted != text:
                self._log_cap_manifest(event, text, compacted)
                text = compacted

        self._final_cache[key] = text
        return text

    def _log_cap_manifest(self, event, original: str, processed: str) -> None:
        if not self.storage:
            return
        head_length = processed.find("\n…[сокращено")
        tail_start = processed.rfind("]…\n")
        removed_start = max(head_length, 0)
        removed_end = tail_start + 3 if tail_start >= 0 else len(original)
        self.storage.log_manifest(
            "",
            event_id=self._event_key(event),
            invocation_id=getattr(event, "invocation_id", "") or "",
            role=self._role_of(event),
            original=original,
            processed=processed,
            removed=[{"start": removed_start, "end": removed_end, "omitted": True}],
            ops=["cap"],
        )

    def _select_events(self, session) -> list:
        events = [event for event in (session.events or []) if self._has_parts(event)]
        window = self.window
        if window is None:
            return events
        return window.select(events, size=self._final_size)

    def _final_size(self, event) -> int:
        window = self.window
        return 0 if window is None else window.count(self._final_text(event))

    def _build_contents(self, events: list) -> list:
        contents: list[Content] = []
        for event in events:
            content = event.content
            if not content or not content.parts:
                continue

            parts: list[Part] = []
            for part in content.parts:
                if part.text:
                    final_text = self._final_text(event)
                    safe_text = self.secret_editor.redact(final_text) if self.secret_editor else final_text
                    parts.append(Part(text=safe_text))
                else:
                    parts.append(part)
            if parts:
                contents.append(Content(role=self._role_of(event), parts=parts))
        return contents

    # ─────────────────────── шаг 4: тон ───────────────────────

    def _apply_tone_hint(self, new_contents: list, callback_context: CallbackContext):
        if not self.analyzer or not self.tone_hints:
            return

        emotion = callback_context.state.get("current_emotion", "neutral")
        tone_hint = self.tone_hints.get(emotion)

        if tone_hint and new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
            part = new_contents[-1].parts[0]
            prefix = f"{part.text}\n\n" if part.text else ""
            part.text = f"{prefix}[Tone hint: {tone_hint}]"

    # ──────────── НОВОЕ: обработка ответов инструментов ────────────

    def _process_tool_responses(self, llm_request: LlmRequest) -> None:
        """Обрабатывает function_response в llm_request.contents (только новые, по ID)."""
        if not self.secret_editor:
            return

        for content in llm_request.contents:
            if not content.parts:
                continue
            for part in content.parts:
                if getattr(part, "function_response", None):
                    resp = part.function_response
                    # Уникальный ID ответа (если нет — генерируем хэш)
                    resp_id = getattr(resp, "id", None)
                    if not resp_id:
                        raw = f"{resp.name}|{json.dumps(resp.response, sort_keys=True, ensure_ascii=False, default=str)}"
                        resp_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                    
                    if resp_id in self._processed_tool_ids:
                        continue
                    
                    self._processed_tool_ids.add(resp_id)
                    
                    response = resp.response
                    if isinstance(response, dict):
                        for key, value in response.items():
                            if isinstance(value, str):
                                # 1. DLP (защита от утечек)
                                safe = self.secret_editor.redact(value)
                                # 2. Жесткий лимит на размер
                                if len(safe) > self.tool_response_max_chars:
                                    safe = safe[:self.tool_response_max_chars] + f"\n...[TRUNCATED: {len(value)} -> {self.tool_response_max_chars}]"
                                response[key] = safe

    # ─────────────────────── сборка ───────────────────────

    def before_model(self, callback_context: CallbackContext, llm_request: LlmRequest):
        session = callback_context.session
        if not session:
            return

        self._ledger_new_events(session, callback_context)

        # Если это повторный вызов после инструмента — обрабатываем только новые tool responses
        if callback_context.state.get(self.flag_key):
            self._process_tool_responses(llm_request)
            return

        self._session_id = session.id

        self._process_last_user_message(session, callback_context)

        selected = self._select_events(session)
        new_contents = self._build_contents(selected)

        self._apply_tone_hint(new_contents, callback_context)

        window = self.window
        if window is not None:
            mode = window.describe()
            total_tokens = sum(window.count(self._final_text(event)) for event in selected)
        else:
            mode = "вся история"
            total_tokens = 0
        print(f"=== КОНТЕКСТ ({mode}; сообщений: {len(new_contents)}; ~{total_tokens} токенов) ===")
        llm_request.contents = new_contents

        callback_context.state[self.flag_key] = True

    def before_tool(self, tool, args, tool_context):
        """Вызывается ДО выполнения инструмента. Проверяет guards и HITL."""
        for guard in self.guards:
            if not guard.check(tool.name, args, {}):
                return {"error": f"Tool '{tool.name}' blocked by security guard."}

        if self.enable_hitl and tool.name in self.hitl_tools:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
            args_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()[:8]
            state_key = f"hitl_approved_{tool.name}_{args_hash}"

            if not tool_context.state.get(state_key):
                return {
                    "status": "confirmation_required",
                    "message": (
                        f"⚠️ HITL: Действие '{tool.name}' требует подтверждения пользователя. "
                        f"Аргументы: {args}. "
                        f"Спросите пользователя, разрешить ли это действие. "
                        f"Если он согласен, вызовите инструмент `approve_action` с параметрами: "
                        f"tool_name='{tool.name}', args_hash='{args_hash}'. "
                        f"После этого вы сможете снова вызвать '{tool.name}'."
                    )
                }

        return None

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
