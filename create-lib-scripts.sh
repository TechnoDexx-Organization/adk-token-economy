#!/bin/bash

echo "🚀 Создаем универсальную библиотеку agent-harness..."

# Создаем структуру папок
mkdir -p agent-harness/src/harness/core
mkdir -p agent-harness/src/harness/modules
mkdir -p agent-harness/src/harness/adapters
cd agent-harness

# 1. Конфигурация пакета
cat << 'EOF' > pyproject.toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-harness"
version = "0.1.0"
description = "Универсальная обвязка для LLM-агентов (экономия, память, эмоции, безопасность)"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "python-dotenv",
]

[project.optional-dependencies]
adk = ["google-adk>=2.13.0", "litellm>=1.84", "transformers>=4.40", "torch"]
nlp = ["transformers>=4.40", "torch"]

[tool.setuptools.packages.find]
where = ["src"]
EOF

# 2. README
cat << 'EOF' > README.md
# Agent Harness
Универсальный конструктор (обвязка) для любых LLM-агентов.
Позволяет независимо подключать сушку текста, скользящее окно, анализ эмоций, ледник (cold storage) и защиту инструментов (ToolGuard).
EOF

# 3. Точка входа
cat << 'EOF' > src/harness/__init__.py
from .core.wrapper import AgentWrapper
from .core.interfaces import PreProcessor, PostProcessor, ToolGuard

# Экспортируем модули для удобства
from .modules.analyzer import EmotionAnalyzer
from .modules.sparsifier import TextSparsifier
from .modules.storage import ColdStorage
from .modules.window import SlidingWindow
from .modules.guard import SafeToolGuard

__all__ = [
    "AgentWrapper",
    "PreProcessor",
    "PostProcessor",
    "ToolGuard",
    "EmotionAnalyzer",
    "TextSparsifier",
    "ColdStorage",
    "SlidingWindow",
    "SafeToolGuard",
]
EOF

# 4. Базовые интерфейсы (Core)
cat << 'EOF' > src/harness/core/interfaces.py
from abc import ABC, abstractmethod

class PreProcessor(ABC):
    @abstractmethod
    def process(self, text: str, context: dict) -> str:
        """Обработка текста ДО отправки в агента."""
        pass

class PostProcessor(ABC):
    @abstractmethod
    def process(self, text: str, context: dict) -> str:
        """Обработка ответа ПОСЛЕ агента."""
        pass

class ToolGuard(ABC):
    @abstractmethod
    def check(self, tool_name: str, args: dict, context: dict) -> bool:
        """Проверка, можно ли вызывать инструмент (True - можно, False - заблокировать)."""
        pass
EOF

# 5. Главный класс обертки (Core)
cat << 'EOF' > src/harness/core/wrapper.py
from typing import Any
from .interfaces import PreProcessor, PostProcessor, ToolGuard

class AgentWrapper:
    def __init__(
        self,
        agent: Any,
        pre_processors: list[PreProcessor] = None,
        post_processors: list[PostProcessor] = None,
        guards: list[ToolGuard] = None,
    ):
        self.agent = agent
        self.pre_processors = pre_processors or []
        self.post_processors = post_processors or []
        self.guards = guards or []

    def process_input(self, text: str, context: dict) -> str:
        """Прогоняет текст через все pre-processors (сушка, окно и т.д.)."""
        processed = text
        for pp in self.pre_processors:
            processed = pp.process(processed, context)
        return processed

    def process_output(self, text: str, context: dict) -> str:
        """Прогоняет ответ через все post-processors."""
        processed = text
        for po in self.post_processors:
            processed = po.process(processed, context)
        return processed

    def invoke(self, text: str, context: dict = None) -> str:
        """Универсальный метод вызова любого агента."""
        if context is None:
            context = {}
        
        safe_input = self.process_input(text, context)
        
        # Предполагаем, что у агента есть метод invoke или generate
        if hasattr(self.agent, "invoke"):
            raw_output = self.agent.invoke(safe_input)
        elif hasattr(self.agent, "generate"):
            raw_output = self.agent.generate(safe_input)
        elif callable(self.agent):
            raw_output = self.agent(safe_input)
        else:
            raise ValueError("Агент должен иметь метод invoke/generate или быть вызываемой функцией.")
        
        return self.process_output(raw_output, context)
EOF

# 6. Модуль: Анализ эмоций
cat << 'EOF' > src/harness/modules/analyzer.py
class EmotionAnalyzer:
    def __init__(self):
        print("⏳ Загрузка модели эмоционального анализа...")
        try:
            from transformers import pipeline
            self.pipeline = pipeline(
                "text-classification",
                model="Aniemore/rubert-tiny2-russian-emotion-detection",
            )
            print("✅ Модель эмоционального анализа загружена")
        except Exception as e:
            self.pipeline = None
            print(f"⚠️ Не удалось загрузить модель: {e}")

    def analyze(self, text: str) -> str:
        if not self.pipeline:
            return "neutral"
        try:
            result = self.pipeline(text, truncation=True, max_length=512)[0]
            return result["label"]
        except Exception:
            return "neutral"
EOF

# 7. Модуль: Сушка
cat << 'EOF' > src/harness/modules/sparsifier.py
import re
from ..core.interfaces import PreProcessor

class TextSparsifier(PreProcessor):
    def __init__(self):
        self.filler_words = [
            r"\bкороче\b", r"\bв общем\b", r"\bкак бы\b", r"\bтипа\b",
            r"\bна самом деле\b", r"\bв принципе\b", r"\bпо сути\b",
            r"\bсобственно\b", r"\bпросто\b", r"\bзначит\b",
        ]

    def process(self, text: str, context: dict) -> str:
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
EOF

# 8. Модуль: Ледник
cat << 'EOF' > src/harness/modules/storage.py
import json
import os
from datetime import datetime, timezone

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
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"⚠️ Ошибка записи в ледник: {e}")
EOF

# 9. Модуль: Скользящее окно
cat << 'EOF' > src/harness/modules/window.py
from ..core.interfaces import PreProcessor

class SlidingWindow(PreProcessor):
    def __init__(self, max_events: int = 10):
        self.max_events = max_events

    def process(self, text: str, context: dict) -> str:
        # В универсальной обвязке окно работает со списком сообщений в context.
        # Здесь заглушка для одиночного текста, но логика сохраняется.
        return text

    def apply_to_history(self, events: list) -> list:
        if self.max_events <= 0:
            return events
        return events[-self.max_events:]
EOF

# 10. Модуль: Защита инструментов (ToolGuard)
cat << 'EOF' > src/harness/modules/guard.py
from ..core.interfaces import ToolGuard

class SafeToolGuard(ToolGuard):
    def __init__(self, allow_list: list = None, deny_list: list = None):
        self.allow_list = allow_list or []
        self.deny_list = deny_list or []

    def check(self, tool_name: str, args: dict, context: dict) -> bool:
        if tool_name in self.deny_list:
            print(f"🚫 Блокировка (deny_list): {tool_name}")
            return False
        if self.allow_list and tool_name not in self.allow_list:
            print(f"🚫 Блокировка (не в allow_list): {tool_name}")
            return False
        return True
EOF

# 11. Адаптер для ADK (Связывает нашу обертку с колбэками ADK)
cat << 'EOF' > src/harness/adapters/adk.py
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai.types import Content, Part
from ..modules.window import SlidingWindow
from ..modules.storage import ColdStorage
from ..modules.analyzer import EmotionAnalyzer
from ..modules.sparsifier import TextSparsifier

class ADKAgentCallbacks:
    """Класс-адаптер, который транслирует логику Harness в колбэки Google ADK."""
    def __init__(
        self,
        analyzer: EmotionAnalyzer = None,
        sparsifier: TextSparsifier = None,
        storage: ColdStorage = None,
        window: SlidingWindow = None,
        tone_hints: dict = None,
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
        if callback_context.state.get(self.flag_key):
            return
        session = callback_context.session
        if not session:
            return

        # 1. ЛЕДНИК
        if self.storage:
            for event in reversed(session.events):
                if event.author == "user":
                    if orig_text := self._event_to_text(event):
                        self.storage.log(session.id, "user", orig_text)
                    break

        # 2. ИСТОРИЯ (Эмоции и Сушка)
        all_valid_events = []
        for event in session.events:
            text = self._event_to_text(event)
            if not text:
                continue
            role = "user" if event.author == "user" else "model"

            if role == "user":
                if self.analyzer:
                    emotion = self.analyzer.analyze(text)
                    callback_context.state["current_emotion"] = emotion
                
                if self.sparsifier:
                    text = self.sparsifier.process(text, {})

            all_valid_events.append((role, text))

        # 3. ОКНО
        if self.window:
            windowed_events = self.window.apply_to_history(all_valid_events)
        else:
            windowed_events = all_valid_events

        new_contents = [Content(role=role, parts=[Part(text=text)]) for role, text in windowed_events]

        # 4. ТОН
        if self.analyzer and self.tone_hints:
            emotion = callback_context.state.get("current_emotion", "neutral")
            tone_hint = self.tone_hints.get(emotion)
            if tone_hint and new_contents and new_contents[-1].role == "user" and new_contents[-1].parts:
                part = new_contents[-1].parts[0]
                prefix = f"{part.text}\n\n" if part.text else ""
                part.text = f"{prefix}[Tone hint: {tone_hint}]"

        llm_request.contents = new_contents
        callback_context.state[self.flag_key] = True

    def after_model(self, callback_context: CallbackContext, llm_response: LlmResponse):
        # ... (код after_model из старой библиотеки) ...
        pass # Заглушка, вставим потом
EOF

echo "✅ Библиотека agent-harness успешно создана!"
echo "📁 Структура:"
find . -type f | sort
