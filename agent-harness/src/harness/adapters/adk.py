import difflib
import hashlib
import json
import logging

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part

from ..core.interfaces import GuardDecision
from ..modules.analyzer import EmotionAnalyzer
from ..modules.canary import CanaryGuard
from ..modules.cost import CostLimitExceeded, CostTracker
from ..modules.memory import FactMemoryProvider, MemoryProvider
from ..modules.secret_editor import SecretEditor
from ..modules.sparsifier import TextSparsifier
from ..modules.storage import ColdStorage
from ..modules.window import SlidingWindow

logger = logging.getLogger(__name__)


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
        memory_provider: MemoryProvider | None = None,
        fact_memory_provider: FactMemoryProvider | None = None,
        canary_guard: CanaryGuard | None = None,
        cost_tracker: CostTracker | None = None,
    ):
        self.analyzer = analyzer
        self.sparsifier = sparsifier
        self.storage = storage
        self.window = window
        self.tone_hints = tone_hints or {}
        self.guards = guards or []
        # Передаем storage в SecretEditor, чтобы аудит работал
        self.secret_editor = secret_editor or SecretEditor(storage=storage)
        self.enable_hitl = enable_hitl
        self.memory_provider = memory_provider
        self.fact_memory_provider = fact_memory_provider
        # Канареечный токен: None => защита выключена (маркер в промпт не вставляется)
        self.canary_guard = canary_guard
        self.cost_tracker = cost_tracker or CostTracker()
        self.flag_key = "_processed"
        self.ledger_key = "_ledger_upto"

        self.hitl_tools = {"run_shell_command", "write_file"}
        self.tool_response_max_chars = 10_000
        self._processed_tool_ids: set[str] = set()

        self._sparse_cache: dict[str, str] = {}
        self._emotion_cache: dict[str, str] = {}
        self._final_cache: dict[str, str] = {}
        self._cache_max_size = 4096

    def _event_to_text(self, event) -> str:
        if not event.content or not event.content.parts:
            return ""
        return "".join(part.text for part in event.content.parts if part.text)

    def _event_key(self, event) -> str:
        if event_id := getattr(event, "id", "") or "":
            return event_id
        raw = f"{getattr(event, 'author', '')}|{getattr(event, 'invocation_id', '')}|{self._event_to_text(event)}"
        return "synthetic-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _cache_key(self, session_id: str, event_key: str) -> str:
        return f"{session_id}:{event_key}"

    def _role_of(self, event) -> str:
        return "user" if getattr(event, "author", "") == "user" else "model"

    @staticmethod
    def _get_opcodes(original: str, processed: str) -> list:
        """Возвращает список opcodes для манифеста."""
        matcher = difflib.SequenceMatcher(None, original, processed, autojunk=False)
        return matcher.get_opcodes()

    def _evict_lru(self, cache: dict):
        while len(cache) > self._cache_max_size:
            oldest_key = next(iter(cache))
            del cache[oldest_key]

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

    def _process_last_user_message(self, session, callback_context: CallbackContext) -> str | None:
        last_user_event = next(
            (
                event
                for event in reversed(session.events or [])
                if getattr(event, "author", "") == "user" and self._event_to_text(event)
            ),
            None,
        )
        if last_user_event is None:
            return None

        original = self._event_to_text(last_user_event)
        event_key = self._event_key(last_user_event)
        cache_key = self._cache_key(session.id, event_key)
        invocation_id = getattr(last_user_event, "invocation_id", "") or ""

        if self.analyzer:
            emotion = self._emotion_cache.get(cache_key)
            if emotion is None:
                emotion = self.analyzer.analyze(original)
                self._evict_lru(self._emotion_cache)
                self._emotion_cache[cache_key] = emotion
            callback_context.state["current_emotion"] = emotion

        processed = original
        if self.sparsifier:
            cached = self._sparse_cache.get(cache_key)
            if cached is None:
                processed = self.sparsifier.process(original, {})
                self._evict_lru(self._sparse_cache)
                self._sparse_cache[cache_key] = processed
                self._final_cache.pop(cache_key, None)
                if self.storage and processed != original:
                    self.storage.log_manifest(
                        session.id,
                        event_id=event_key,
                        invocation_id=invocation_id,
                        role="user",
                        original=original,
                        processed=processed,
                        opcodes=self._get_opcodes(original, processed),
                        ops=["sparsify"],
                    )
            else:
                processed = cached
        return processed

    def _final_text(self, session_id: str, event) -> str:
        event_key = self._event_key(event)
        cache_key = self._cache_key(session_id, event_key)
        cached = self._final_cache.get(cache_key)
        if cached is not None:
            return cached

        text = self._event_to_text(event)
        if self._role_of(event) == "user":
            text = self._sparse_cache.get(cache_key, text)

        window = self.window
        if window is not None:
            compacted = window.compact(text)
            if compacted != text:
                self._log_cap_manifest(session_id, event, text, compacted)
                text = compacted

        self._evict_lru(self._final_cache)
        self._final_cache[cache_key] = text
        return text

    def _log_cap_manifest(self, session_id: str, event, original: str, processed: str) -> None:
        if not self.storage:
            return
        self.storage.log_manifest(
            session_id,
            event_id=self._event_key(event),
            invocation_id=getattr(event, "invocation_id", "") or "",
            role=self._role_of(event),
            original=original,
            processed=processed,
            opcodes=self._get_opcodes(original, processed),
            ops=["cap"],
        )

    def _select_events(self, session) -> list:
        events = [event for event in (session.events or []) if self._has_parts(event)]
        window = self.window
        if window is None:
            return events
        return window.select(events, size=lambda e: self._final_size(session.id, e))

    def _final_size(self, session_id: str, event) -> int:
        window = self.window
        return 0 if window is None else window.count(self._final_text(session_id, event))

    def _build_contents(self, session_id: str, events: list) -> list:
        contents: list[Content] = []
        for event in events:
            content = event.content
            if not content or not content.parts:
                continue
            parts: list[Part] = []
            final_text = self._final_text(session_id, event)
            if self.canary_guard and self.canary_guard.check_incoming(final_text, session_id=session_id):
                # Во входящем тексте канареечный маркер — это провокация/инъекция.
                # Текст в модель не уходит, заменяется заглушкой.
                final_text = "[BLOCKED: обнаружен канареечный маркер в запросе — вероятная prompt injection]"
            safe_text = (
                self.secret_editor.redact(
                    final_text,
                    session_id=session_id,
                    event_id=self._event_key(event),
                    invocation_id=getattr(event, "invocation_id", ""),
                    role=self._role_of(event),
                )
                if self.secret_editor
                else final_text
            )

            is_first_text = True
            for part in content.parts:
                if part.text:
                    if is_first_text:
                        parts.append(Part(text=safe_text))
                        is_first_text = False
                else:
                    parts.append(part)
            if parts:
                contents.append(Content(role=self._role_of(event), parts=parts))
        return contents

    def _apply_tone_hint(self, new_contents: list, callback_context: CallbackContext):
        if not self.analyzer or not self.tone_hints:
            return
        emotion = callback_context.state.get("current_emotion", "neutral")
        tone_hint = self.tone_hints.get(emotion)
        if tone_hint and new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
            part = new_contents[-1].parts[0]
            prefix = f"{part.text}\n\n" if part.text else ""
            new_part = Part(text=f"{prefix}[Tone hint: {tone_hint}]")
            new_parts = [new_part] + new_contents[-1].parts[1:]
            new_contents[-1] = Content(role="user", parts=new_parts)

    def _process_tool_responses(self, session_id: str, contents: list[Content]) -> None:
        if not self.secret_editor:
            return
        for content in contents:
            if not content.parts:
                continue
            for idx, part in enumerate(content.parts):
                resp = getattr(part, "function_response", None)
                if resp is not None:
                    if not (resp_id := getattr(resp, "id", None)):
                        raw = (
                            f"{resp.name}|{json.dumps(resp.response, sort_keys=True, ensure_ascii=False, default=str)}"
                        )
                        resp_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
                    cache_key = self._cache_key(session_id, resp_id)
                    if cache_key in self._processed_tool_ids:
                        continue
                    if len(self._processed_tool_ids) > self._cache_max_size:
                        self._processed_tool_ids.clear()
                    self._processed_tool_ids.add(cache_key)
                    response = resp.response
                    if isinstance(response, dict):
                        new_response = dict(response)
                        changed = False
                        for key, value in new_response.items():
                            if isinstance(value, str):
                                safe = self.secret_editor.redact(
                                    value, session_id=session_id, event_id="tool_response", role="tool"
                                )
                                if len(safe) > self.tool_response_max_chars:
                                    safe = f"{safe[: self.tool_response_max_chars]}\n...[TRUNCATED: {len(value)} -> {self.tool_response_max_chars}]"
                                if safe != value:
                                    new_response[key] = safe
                                    changed = True
                        if changed:
                            new_resp = resp.model_copy(update={"response": new_response})
                            new_part = part.model_copy(update={"function_response": new_resp})
                            content.parts[idx] = new_part

    def _apply_memory_search(self, session_id: str, new_contents: list) -> None:
        if not new_contents:
            return
        if new_contents[-1].role == "user" and new_contents[-1].parts:
            part = new_contents[-1].parts[0]
            if query := (getattr(part, "text", "") or ""):
                mem_text = ""
                if self.memory_provider:
                    if memories := self.memory_provider.search(query, limit=3):
                        joined = "\n".join(f"- {m}" for m in memories)
                        mem_text += f"\n\n[Relevant context from past conversations]:\n{joined}"
                        logger.info("Найдено %d фрагментов в памяти.", len(memories))
                if self.fact_memory_provider:
                    if facts := self.fact_memory_provider.search(query, limit=5):
                        joined_facts = "\n".join(f"- {f}" for f in facts)
                        mem_text += f"\n\n[Known facts about user]:\n{joined_facts}"
                        logger.info("Найдено %d фактов о пользователе.", len(facts))
                if mem_text:
                    safe_mem_text = (
                        self.secret_editor.redact(
                            mem_text, session_id=session_id, event_id="memory_context", role="system"
                        )
                        if self.secret_editor
                        else mem_text
                    )
                    new_part = Part(text=query + safe_mem_text)
                    new_parts = [new_part] + new_contents[-1].parts[1:]
                    new_contents[-1] = Content(role="user", parts=new_parts)

    def before_model(self, callback_context: CallbackContext, llm_request: LlmRequest):
        session = callback_context.session
        if not session:
            return
        self._ledger_new_events(session, callback_context)
        is_tool_step = callback_context.state.get(self.flag_key, False)
        if not is_tool_step:
            self._process_last_user_message(session, callback_context)
        selected = self._select_events(session)
        new_contents = self._build_contents(session.id, selected)
        self._process_tool_responses(session.id, new_contents)
        if not is_tool_step:
            self._apply_memory_search(session.id, new_contents)
            self._apply_tone_hint(new_contents, callback_context)
        window = self.window
        if window is not None:
            mode = window.describe()
            total_tokens = sum(window.count(self._final_text(session.id, event)) for event in selected)
        else:
            mode = "вся история"
            total_tokens = 0
        logger.info(
            "Контекст (%s): сообщений=%d, токенов~%d",
            mode,
            len(new_contents),
            total_tokens,
            extra={
                "kind": "context_metrics",
                "mode": mode,
                "messages": len(new_contents),
                "approx_tokens": total_tokens,
            },
        )
        llm_request.contents = new_contents
        callback_context.state[self.flag_key] = True
        callback_context.state["model_name"] = llm_request.model

    def before_tool(self, tool, args, tool_context):
        force_hitl = False
        is_safe = False
        for guard in self.guards:
            decision = guard.check(tool.name, args, {})
            if decision == GuardDecision.DENY:
                return {"error": f"Tool '{tool.name}' blocked by security guard."}
            elif decision == GuardDecision.REQUIRE_HITL:
                force_hitl = True
            elif decision == GuardDecision.ALLOW:
                is_safe = True

        needs_hitl = False
        if force_hitl:
            needs_hitl = True
        elif not is_safe and self.enable_hitl and tool.name in self.hitl_tools:
            needs_hitl = True

        if needs_hitl:
            args_str = json.dumps(args, sort_keys=True, ensure_ascii=False)
            args_hash = hashlib.md5(args_str.encode("utf-8")).hexdigest()[:8]
            state_key = f"hitl_approved_{tool.name}_{args_hash}"
            state = getattr(tool_context, "state", {})
            if not state.get(state_key):
                return {
                    "status": "confirmation_required",
                    "message": (
                        f"⚠️ HITL: Действие '{tool.name}' требует подтверждения пользователя. "
                        f"Аргументы: {args}. "
                        f"Спросите пользователя, разрешить ли это действие. "
                        f"Если он согласен, вызовите инструмент `approve_action` с параметрами: "
                        f"tool_name='{tool.name}', args_hash='{args_hash}'. "
                        f"После этого вы сможете снова вызвать '{tool.name}'."
                    ),
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
        has_function_call = any(part.function_call is not None for part in llm_response.content.parts)
        if reply_text := "".join(part.text for part in llm_response.content.parts if part.text):
            if self.canary_guard and self.canary_guard.check_outgoing(
                reply_text, session_id=getattr(callback_context.session, "id", "")
            ):
                # Маркер в ответе модели => системный промпт утёк наружу.
                callback_context.state["canary_leak"] = True
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Ответ модели:\n%s\n%s", reply_text, "=" * 20)
        if llm_response.usage_metadata:
            cached = getattr(llm_response.usage_metadata, "cached_content_token_count", 0)
            prompt = getattr(llm_response.usage_metadata, "prompt_token_count", 0)
            total = getattr(llm_response.usage_metadata, "total_token_count", 0)
            logger.info(
                "Метрики: prompt=%s, cached=%s",
                prompt,
                cached,
                extra={"kind": "usage_metrics", "prompt_tokens": prompt, "cached_tokens": cached},
            )
            try:
                model_name = callback_context.state.get("model_name", "unknown")
                self.cost_tracker.track(
                    model_name=model_name,
                    prompt_tokens=prompt,
                    cached_tokens=cached,
                    total_tokens=total,
                )
            except CostLimitExceeded:
                # Бюджет исчерпан: помечаем состояние и останавливаем исполнение.
                callback_context.state["cost_limit_exceeded"] = True
                llm_response.content = Content(
                    role="model",
                    parts=[Part(text=f"⛔ {CostLimitExceeded(self.cost_tracker.limit)}")],
                )
                llm_response.error_code = "cost_limit_exceeded"
                llm_response.error_message = str(CostLimitExceeded(self.cost_tracker.limit))
                return
        if not has_function_call:
            callback_context.state[self.flag_key] = False
            logger.debug("after_model: финал, флаг снят")

    def _handle_empty_response(self, callback_context: CallbackContext, llm_response: LlmResponse):
        callback_context.state[self.flag_key] = False
        error_count = callback_context.state.get("error_count", 0) + 1
        callback_context.state["error_count"] = error_count
        reason = f"Сбой API: {llm_response.error_code}" if llm_response.error_code else "Модель вернула пустоту"
        logger.warning("%s. Сбой №%d.", reason, error_count)
        if session := callback_context.session:
            for event in reversed(session.events):
                if event.author == "user":
                    if text := self._event_to_text(event):
                        llm_response.content = Content(role="model", parts=[Part(text=text)])
                        llm_response.error_code = None
                        llm_response.error_message = None
                    break
