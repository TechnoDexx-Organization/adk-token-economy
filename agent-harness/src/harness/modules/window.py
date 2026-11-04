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
