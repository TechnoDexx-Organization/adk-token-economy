import re
from ..core.interfaces import PreProcessor

class TextSparsifier(PreProcessor):
    def __init__(self):
        self.filler_words = [
            r"\bкороче\b", r"\bв общем\b", r"\bкак бы\b", r"\bтипа\b",
            r"\bна самом деле\b", r"\bв принципе\b", r"\bпо сути\b",
            r"\bсобственно\b", r"\bпросто\b", r"\bзначит\b",
        ]

    def process(self, text: str, context: dict) -> str:
        for word in self.filler_words:
            text = re.sub(word, "", text, flags=re.IGNORECASE)
        text = (
            text.replace(" потому что ", ". ")
            .replace(" так как ", ". ")
            .replace(" и теперь ", ". ")
        )
        text = text.replace(", и ", ". ").replace(", а ", ". ")
        text = text.replace(",.", ".").replace(", .", ".")
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            text = text[0].upper() + text[1:]
        return text
