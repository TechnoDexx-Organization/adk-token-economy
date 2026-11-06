"""Общие помощники для тестов: события ADK без сети и без модели."""
from google.adk.events import Event
from google.genai.types import Content, Part


def make_event(author: str, text: str, invocation_id: str = "", parts=None) -> Event:
    """Собирает событие ADK так, как его отдаёт настоящий Runner."""
    if parts is None:
        parts = [Part(text=text)]
    role = "user" if author == "user" else "model"
    return Event(
        author=author,
        invocation_id=invocation_id,
        content=Content(role=role, parts=parts),
    )


def text_of(event: Event) -> str:
    if not event.content or not event.content.parts:
        return ""
    return "".join(part.text for part in event.content.parts if part.text)
