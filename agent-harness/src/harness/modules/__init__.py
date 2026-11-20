# agent-harness/src/harness/modules/__init__.py
"""
Пакет модулей остаётся без импортов (P1-9): сам пакет `harness.modules`
тяжёлые зависимости не тянет. Публичный API — в `harness/__init__.py`
(ленивые импорты), либо импортируйте нужный модуль напрямую:
`from harness.modules.window import SlidingWindow`.
"""
