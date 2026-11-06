import re

from ..core.interfaces import PreProcessor

# «Вода», которую можно удалять без потери смысла: вводные слова, связки,
# частицы-затычки. Порядок важен: длинные словосочетания вырезаются раньше
# коротких, иначе «в общем» распадётся на «общем».
SAFE_FILLERS = (
    r"\bв общем(?:-то)?\b",
    r"\bесли честно\b",
    r"\bчестно говоря\b",
    r"\bна самом деле\b",
    r"\bв принципе\b",
    r"\bпо сути(?: дела)?\b",
    r"\bтак сказать\b",
    r"\bтак вот\b",
    r"\bвообще говоря\b",
    r"\bвообще-то\b",
    r"\bвроде бы\b",
    r"\bкак бы\b",
    r"\bпо-моему\b",
    r"\bмежду прочим\b",
    r"\bсобственно(?: говоря)?\b",
    r"\bкороче(?: говоря)?\b",
    r"\bкак-то\b",
    r"\bтипа\b",
    r"\bмол\b",
    r"\bдескать\b",
    r"\bпросто\b",
    r"\bпросто-таки\b",
    r"\bбуквально\b",
    r"\bреально\b",
    r"\bфактически\b",
    r"\bзначит\b",
    r"\bкстати\b",
    r"\bну\b",
)

# Слова, которые часто «лишние», но иногда несут смысл (следствие, противопоставление).
# По умолчанию не удаляются: экономия не стоит потери логики.
RISKY_FILLERS = (
    r"\bтаким образом\b",
    r"\bкроме того\b",
    r"\bболее того\b",
    r"\bтем не менее\b",
    r"\bоднако\b",
    r"\bследовательно\b",
    r"\bсоответственно\b",
    r"\bитак\b",
    r"\bнаверное\b",
    r"\bвозможно\b",
    r"\bкажется\b",
)


class TextSparsifier(PreProcessor):
    """
    Сушка текста: убирает слова-связки и «воду», не трогая смысл.

    Что делает:
      * вырезает вводные слова и частицы-затычки из `SAFE_FILLERS`;
      * схлопывает соседние повторы («очень очень» → «очень»);
      * схлопывает дословно повторённые предложения;
      * наводит порядок в пунктуации и пробелах.

    Чего не делает: не пересказывает, не меняет порядок слов, не выкидывает
    факты. Всё вырезанное попадает в манифест ледника — обратимо.

    `strict` включает удаление `RISKY_FILLERS` (следствие, противопоставление).
    """

    def __init__(
        self,
        fillers: tuple[str, ...] | None = None,
        strict: bool = False,
        dedupe: bool = True,
        keep_causal: bool = False,
    ):
        patterns = tuple(fillers) if fillers is not None else SAFE_FILLERS
        if strict:
            # Осознанный размен: режем и смысловые связки — включать только явно.
            patterns = patterns + RISKY_FILLERS

        self.filler_words = patterns
        self.strict = strict
        self.dedupe = dedupe
        self.keep_causal = keep_causal

    def process(self, text: str, context: dict) -> str:
        if not text:
            return text

        for pattern in self.filler_words:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # «потому что / так как»: по умолчанию схлопываются в точку.
        # keep_causal=True оставляет маркер причины (→), чтобы логика не терялась.
        separator = " → " if self.keep_causal else ". "
        text = (
            text.replace(" потому что ", separator)
            .replace(" так как ", separator)
            .replace(" из-за того что ", separator)
            .replace(" и теперь ", ". ")
        )
        text = text.replace(", и ", ". ").replace(", а ", ". ")

        if self.dedupe:
            text = self._collapse_repeated_words(text)
            text = self._collapse_repeated_sentences(text)

        text = self._tidy_punctuation(text)
        return text

    @staticmethod
    def _collapse_repeated_words(text: str) -> str:
        """«очень очень длинный» → «очень длинный»."""
        return re.sub(r"\b(\w+)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

    @staticmethod
    def _collapse_repeated_sentences(text: str) -> str:
        """Одинаковые соседние предложения схлопываются в одно."""
        pattern = re.compile(r"([^.!?]{15,}?[.!?])(?:\s*\1\s*)+", re.IGNORECASE)
        return pattern.sub(r"\1", text)

    @staticmethod
    def _tidy_punctuation(text: str) -> str:
        """Убирает следы удалений: двойные запятые, пробелы перед знаками."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\s*,\s*,+", ",", text)
        text = re.sub(r",\s*([.!?;:])", r"\1", text)
        text = re.sub(r"([.!?;:])(?:\s*[.!?,;:])+", r"\1", text)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"^[\s,;:.!?]+", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        # Заглавная буква после конца предложения: «Я устал. долго» → «Я устал. Долго».
        text = re.sub(r"([.!?]\s+)(\w)", lambda m: m.group(1) + m.group(2).upper(), text)

        if text:
            text = text[0].upper() + text[1:]
        return text
