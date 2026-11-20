# agent-harness/src/harness/__init__.py
"""
Ленивые импорты публичного API (PEP 562).

Прямые импорты здесь тянули google-adk, litellm и transformers на
`import harness` — даже когда нужен только SlidingWindow. Теперь модуль
грузится в момент первого обращения к его имени.
"""

import logging
from importlib import import_module

# Библиотека не настраивает логирование за приложение: только NullHandler,
# чтобы библиотека ничего не печатала в консумерский вывод (P2-20).
logging.getLogger(__name__).addHandler(logging.NullHandler())

# имя публичного символа -> модуль внутри пакета, где он живёт
_EXPORTS: dict[str, str] = {
    "EmotionAnalyzer": ".modules.analyzer",
    "CanaryGuard": ".modules.canary",
    "build_canary_marker": ".modules.canary",
    "build_canary_token": ".modules.canary",
    "FileTools": ".modules.file_tools",
    "SafeToolGuard": ".modules.guard",
    "ShellCommandGuard": ".modules.guard",
    "build_llm_model": ".modules.llm_factory",
    "MemoryProvider": ".modules.memory",
    "FileMemoryProvider": ".modules.memory",
    "HttpMemoryProvider": ".modules.memory",
    "FactMemoryProvider": ".modules.memory",
    "build_retry_config": ".modules.retry",
    "log_retry_attempt": ".modules.retry",
    "SecretEditor": ".modules.secret_editor",
    "build_safe_skill_toolset": ".modules.skills",
    "is_safe_url": ".modules.ssrf",
    "find_blocked_urls": ".modules.ssrf",
    "TextSparsifier": ".modules.sparsifier",
    "ColdStorage": ".modules.storage",
    "TokenCounter": ".modules.tokens",
    "SlidingWindow": ".modules.window",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path, __name__), name)
    globals()[name] = value  # повторные обращения идут через обычный поиск по модулю
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
