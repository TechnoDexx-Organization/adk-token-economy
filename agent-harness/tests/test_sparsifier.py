"""Сушка: убирает воду, не трогает смысл, обратима через манифест."""
from harness.modules.sparsifier import TextSparsifier


def test_removes_fillers():
    sparsifier = TextSparsifier()

    result = sparsifier.process("Ну, в общем, короче, я на самом деле просто устал.", {})

    assert result == "Я устал."


def test_keeps_meaningful_words():
    sparsifier = TextSparsifier()

    result = sparsifier.process("нужно сделать отчёт к пятнице", {})

    # «нужно» не должно пострадать от удаления частицы «ну»
    assert "нужно" in result.lower()
    assert "отчёт" in result


def test_collapses_repeated_words():
    sparsifier = TextSparsifier()

    result = sparsifier.process("очень очень длинный текст", {})

    assert result == "Очень длинный текст"


def test_causal_kept_when_requested():
    sparsifier = TextSparsifier(keep_causal=True)

    result = sparsifier.process("Я устал, потому что долго работал", {})

    assert "→" in result


def test_strict_mode_removes_risky_fillers():
    text = "Было тяжело, однако мы справились"

    assert "однако" in TextSparsifier().process(text, {})
    assert "однако" not in TextSparsifier(strict=True).process(text, {})
