from abc import ABC, abstractmethod
from enum import Enum


class GuardDecision(str, Enum):
    """Строгий тип для решений Guard'а."""
    ALLOW = "allow"          # Разрешено, можно выполнять
    DENY = "deny"            # Запрещено, блокируем
    REQUIRE_HITL = "require_hitl"  # Серая зона, нужно подтверждение


class PreProcessor(ABC):
    @abstractmethod
    def process(self, text: str, context: dict) -> str:
        """Обработка текста ДО отправки в агента."""

class PostProcessor(ABC):
    @abstractmethod
    def process(self, text: str, context: dict) -> str:
        """Обработка ответа ПОСЛЕ агента."""

class ToolGuard(ABC):
    @abstractmethod
    def check(self, tool_name: str, args: dict, context: dict) -> GuardDecision:
        """Проверка, можно ли вызывать инструмент."""