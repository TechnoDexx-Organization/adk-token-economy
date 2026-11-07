"""Оценка токенов для окон и потолков — без обращения к провайдеру."""


class TokenCounter:
    """
    Счётчик токенов на базе tiktoken (cl100k_base).

    Точное совпадение с токенизатором DeepSeek не гарантируется, но для
    окон и потолков нужна стабильная оценка, а не биллинговая точность.
    Если tiktoken недоступен, работает грубая оценка по символам.
    """

    def __init__(self, encoding_name: str = "cl100k_base", chars_per_token: float = 3.0):
        self.chars_per_token = chars_per_token
        self._encoding = None
        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception:  # noqa: BLE001 — отсутствие tiktoken не должно ронять агента
            self._encoding = None
        self._cache: dict[str, int] = {}

    @property
    def exact(self) -> bool:
        """True, если считаем настоящим токенизатором, а не по символам."""
        return self._encoding is not None

    def encode(self, text: str) -> list[int] | None:
        """Токены текста; None, если токенизатор недоступен."""
        if self._encoding is None:
            return None
        return self._encoding.encode(text, disallowed_special=())

    def decode(self, ids: list[int]) -> str:
        """Обратное преобразование токенов в текст."""
        if self._encoding is None:
            raise RuntimeError("decode вызван без токенизатора")
        return self._encoding.decode(ids)

    def count(self, text: str) -> int:
        if not text:
            return 0
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        if self._encoding is not None:
            tokens = len(self._encoding.encode(text, disallowed_special=()))
        else:
            tokens = max(1, int(len(text) / self.chars_per_token))

        if len(self._cache) < 4096:
            self._cache[text] = tokens
        return tokens
