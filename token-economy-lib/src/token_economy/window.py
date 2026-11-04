class SlidingWindow:
    def __init__(self, max_events: int = 10):
        self.max_events = max_events

    def apply(self, events: list) -> list:
        """Обрезает историю до последних N событий."""
        if self.max_events <= 0:
            return events
        return events[-self.max_events:]
