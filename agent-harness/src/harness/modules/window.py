from ..core.interfaces import PreProcessor
from .tokens import TokenCounter


class SlidingWindow(PreProcessor):
    """
    Окно контекста. Три независимых рычага, которые можно включать вместе:

    * `max_messages` — оставить последние N **сообщений** (событий ADK с текстом).
      Режим, в котором снимался замер "окно 10 сообщений".
    * `max_turns` — оставить последние N **ходов** целиком. Ход определяется
      `event.invocation_id`: реплика пользователя + всё, что модель и
      инструменты сделали в ответ.
    * `max_tokens` — урезать окно до **бюджета в токенах**, выбрасывая самые
      старые сообщения. Работает поверх сообщений или ходов.

    Отдельно — `max_message_tokens`: потолок на **одно** сообщение. Длинный
    текст сокращается по принципу голова+хвост, середина заменяется маркером.
    Именно этот потолок делает экономию растущей вместе с размером сообщений:
    чем длиннее реплика, тем больше от неё отрезается.

    Все три параметра действуют по цепочке: сначала сообщения/ходы, затем
    бюджет токенов; потолок на сообщение применяется к тексту до сборки запроса.
    """

    ELISION_TEMPLATE = "\n…[сокращено {tokens} токенов]…\n"

    def __init__(
        self,
        max_turns: int | None = None,
        max_messages: int | None = None,
        max_events: int | None = None,
        max_tokens: int | None = None,
        max_message_tokens: int | None = None,
        counter: TokenCounter | None = None,
        head_ratio: float = 0.6,
    ):
        # max_events — старое имя, оставлено для обратной совместимости:
        # это тот же режим "по сообщениям".
        self.max_turns = max_turns
        self.max_messages = max_messages if max_messages is not None else max_events
        self.max_tokens = max_tokens
        self.max_message_tokens = max_message_tokens
        self.counter = counter or TokenCounter()
        self.head_ratio = min(max(head_ratio, 0.0), 1.0)

        if self.max_messages is None and self.max_turns is None and self.max_tokens is None:
            self.max_turns = 10

    def process(self, text: str, context: dict) -> str:
        # Одиночный текст окно не сокращает — логика в select() и compact().
        return text

    def count(self, text: str) -> int:
        return self.counter.count(text)

    # ───────────────────────── ходы ─────────────────────────

    def group_by_turns(self, events: list) -> list[list]:
        """
        Разбивает список событий ADK на ходы.

        Ход начинается с события, у которого новый `invocation_id`, либо с
        user-события, если предыдущий ход уже содержит user-событие (страховка
        на случай пустых invocation_id в тестах и сторонних фреймворках).
        """
        turns: list[list] = []
        for event in events:
            invocation_id = getattr(event, "invocation_id", "") or ""
            author = getattr(event, "author", "") or ""

            if not turns:
                turns.append([event])
                continue

            current = turns[-1]
            current_invocation = getattr(current[0], "invocation_id", "") or ""
            current_has_user = any(
                (getattr(item, "author", "") or "") == "user" for item in current
            )
            changed_invocation = bool(invocation_id) and bool(current_invocation) and (
                invocation_id != current_invocation
            )
            repeated_user = author == "user" and (
                current_has_user or not (invocation_id or current_invocation)
            )

            if changed_invocation or repeated_user:
                turns.append([event])
            else:
                turns[-1].append(event)
        return turns

    # ───────────────────────── выбор ─────────────────────────

    def select(self, events: list, size=None) -> list:
        """
        Возвращает события, попадающие в окно.

        `size` — функция event -> число токенов (нужна для `max_tokens`).
        Без неё бюджет считается по тексту события.
        """
        selected = list(events)

        if self.max_messages is not None and self.max_messages > 0:
            selected = selected[-self.max_messages:]
        elif self.max_turns is not None and self.max_turns > 0:
            turns = self.group_by_turns(selected)
            selected = [event for turn in turns[-self.max_turns:] for event in turn]

        if self.max_tokens is not None and self.max_tokens > 0 and selected:
            selected = self._trim_to_budget(selected, size)

        return selected

    def _trim_to_budget(self, events: list, size=None) -> list:
        """Оставляет с конца столько событий, сколько влезает в бюджет."""
        budget = self.max_tokens
        if budget is None or budget <= 0:
            return list(events)
        measure = size or self._default_size
        kept: list = []
        total = 0
        for event in reversed(events):
            cost = measure(event)
            if kept and total + cost > budget:
                break
            kept.append(event)
            total += cost
        kept.reverse()
        return kept

    def _default_size(self, event) -> int:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            return 0
        return sum(self.count(part.text) for part in parts if getattr(part, "text", None))

    # ───────────────────── потолок на сообщение ─────────────────────

    def compact(self, text: str) -> str:
        """
        Сокращает одно сообщение до `max_message_tokens` (голова + хвост).

        Экономия растёт вместе с размером сообщения: пока текст короче потолка,
        он не меняется вообще; чем длиннее — тем больше вырезается.
        """
        if not text or not self.max_message_tokens or self.max_message_tokens <= 0:
            return text

        total = self.count(text)
        if total <= self.max_message_tokens:
            return text

        head_tokens = int(self.max_message_tokens * self.head_ratio)
        tail_tokens = self.max_message_tokens - head_tokens

        head = self._take_tokens(text, head_tokens, from_start=True)
        tail = self._take_tokens(text, tail_tokens, from_start=False)
        removed = total - self.count(head) - self.count(tail)
        marker = self.ELISION_TEMPLATE.format(tokens=max(removed, 0))
        return f"{head}{marker}{tail}"

    def _take_tokens(self, text: str, limit: int, *, from_start: bool) -> str:
        """Берёт примерно `limit` токенов с начала или с конца строки."""
        if limit <= 0:
            return ""
        ids = self.counter.encode(text)
        if ids is not None:
            if len(ids) <= limit:
                return text
            chunk = ids[:limit] if from_start else ids[-limit:]
            return self.counter.decode(chunk)

        # Без токенизатора — режем по символам.
        chars = int(limit * self.counter.chars_per_token)
        if len(text) <= chars:
            return text
        return text[:chars] if from_start else text[-chars:]

    def describe(self) -> str:
        """Человеческое описание режима — для логов эксперимента."""
        parts = []
        if self.max_messages is not None:
            parts.append(f"max_messages={self.max_messages}")
        elif self.max_turns is not None:
            parts.append(f"max_turns={self.max_turns}")
        if self.max_tokens:
            parts.append(f"max_tokens={self.max_tokens}")
        if self.max_message_tokens:
            parts.append(f"max_message_tokens={self.max_message_tokens}")
        return "по " + (", ".join(parts) if parts else "всей истории")

    def apply_to_history(self, events: list) -> list:
        """Совместимость со старым именем: то же, что select()."""
        return self.select(events)
