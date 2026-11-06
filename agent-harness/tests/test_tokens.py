"""Счётчик токенов: оценка работает и без tiktoken."""
from harness.modules.tokens import TokenCounter


def test_counts_grow_with_text():
    counter = TokenCounter()

    short = counter.count("привет")
    long = counter.count("привет " * 50)

    assert 0 < short < long


def test_empty_text_is_zero():
    assert TokenCounter().count("") == 0


def test_fallback_without_tiktoken(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tiktoken":
            raise ImportError("нет tiktoken")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    counter = TokenCounter()

    assert counter.exact is False
    assert counter.count("десять симв") > 0
