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
