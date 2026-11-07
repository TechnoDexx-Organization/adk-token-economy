from typing import Any

from .interfaces import PostProcessor, PreProcessor, ToolGuard


class AgentWrapper:
    def __init__(
        self,
        agent: Any,
        pre_processors: list[PreProcessor] | None = None,
        post_processors: list[PostProcessor] | None = None,
        guards: list[ToolGuard] | None = None,
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

    def invoke(self, text: str, context: dict | None = None) -> str:
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
