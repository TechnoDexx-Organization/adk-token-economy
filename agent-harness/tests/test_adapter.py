"""Адаптер ADK: четыре независимых шага и обратимость."""
import json

from conftest import make_event
from google.genai.types import FunctionCall, Part

from harness.adapters.adk import ADKAgentCallbacks
from harness.modules.storage import ColdStorage
from harness.modules.window import SlidingWindow


class CountingAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, text: str) -> str:
        self.calls += 1
        return "neutral"


class CountingSparsifier:
    def __init__(self):
        self.calls = 0

    def process(self, text: str, context: dict) -> str:
        self.calls += 1
        return text.replace("в общем, ", "")


class FakeSession:
    def __init__(self, events):
        self.id = "session-1"
        self.events = events


class FakeContext:
    def __init__(self, session):
        self.session = session
        self.state = {}


class FakeRequest:
    def __init__(self):
        self.contents: list = []


def build(events, window=None, storage=None, analyzer=None, sparsifier=None):
    callbacks = ADKAgentCallbacks(
        analyzer=analyzer,
        sparsifier=sparsifier,
        storage=storage,
        window=window or SlidingWindow(max_messages=10),
        tone_hints={},
        guards=[],
    )
    context = FakeContext(FakeSession(events))
    return callbacks, context


def test_analyzer_and_sparsifier_run_once_per_new_message(tmp_path):
    events = [make_event("user", "в общем, " + "слово " * 50, "inv1")]
    for turn in range(2, 6):
        events.append(make_event("model", f"ответ {turn}", f"inv{turn}"))
        events.append(make_event("user", f"в общем, вопрос {turn}", f"inv{turn}-u"))

    analyzer = CountingAnalyzer()
    sparsifier = CountingSparsifier()
    callbacks, context = build(
        events, storage=ColdStorage(str(tmp_path)), analyzer=analyzer, sparsifier=sparsifier
    )

    callbacks.before_model(context, FakeRequest())

    # История длиннее одного сообщения, но обработка — только последнего.
    assert analyzer.calls == 1
    assert sparsifier.calls == 1


def read_ledger(tmp_path):
    with open(tmp_path / "cold_storage.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_ledger_records_user_and_model_before_processing(tmp_path):
    events = [
        make_event("user", "в общем, привет", "inv1"),
        make_event("model", "здравствуй", "inv1"),
    ]
    storage = ColdStorage(str(tmp_path))
    callbacks, context = build(events, storage=storage, sparsifier=CountingSparsifier())

    callbacks.before_model(context, FakeRequest())

    entries = read_ledger(tmp_path)
    kinds = {entry["kind"] for entry in entries}

    assert kinds == {"user_message", "model_message", "manifest"}
    user_entry = next(entry for entry in entries if entry["kind"] == "user_message")
    assert user_entry["text"] == "в общем, привет"  # оригинал, не подсушенный


def test_window_applies_to_contents_and_sparsification_is_reversible(tmp_path):
    events = []
    for turn in range(1, 8):
        events.append(make_event("user", f"в общем, вопрос {turn}", f"inv{turn}"))
        events.append(make_event("model", f"ответ {turn}", f"inv{turn}"))

    storage = ColdStorage(str(tmp_path))
    callbacks, context = build(
        events, storage=storage, window=SlidingWindow(max_messages=4),
        sparsifier=CountingSparsifier(),
    )
    request = FakeRequest()
    callbacks.before_model(context, request)

    assert len(request.contents) == 4
    assert request.contents[-1].parts[0].text == "ответ 7"

    manifests = [entry for entry in read_ledger(tmp_path) if entry["kind"] == "manifest"]
    manifest = manifests[-1]
    restored = manifest["processed"]
    for span in sorted(manifest["removed"], key=lambda item: item["start"], reverse=True):
        restored = restored[: span["start"]] + span["text"] + restored[span["start"] :]

    assert restored == manifest["original"]


def test_long_message_is_capped_and_manifested(tmp_path):
    long_text = "очень длинный текст " * 600
    events = [make_event("user", long_text, "inv1")]
    storage = ColdStorage(str(tmp_path))
    callbacks, context = build(
        events,
        storage=storage,
        window=SlidingWindow(max_messages=10, max_message_tokens=200),
        sparsifier=CountingSparsifier(),
    )
    request = FakeRequest()
    callbacks.before_model(context, request)

    sent = request.contents[0].parts[0].text
    assert "сокращено" in sent
    assert len(sent) < len(long_text)

    manifests = [entry for entry in read_ledger(tmp_path) if entry["kind"] == "manifest"]
    assert any(manifest["ops"] == ["cap"] for manifest in manifests)


def test_function_call_parts_are_preserved(tmp_path):
    call_part = Part(function_call=FunctionCall(name="run_shell_command", args={"cmd": "ls"}))
    events = [
        make_event("user", "сделай что-нибудь", "inv1"),
        make_event("model", "", "inv1", parts=[call_part]),
    ]
    callbacks, context = build(events, storage=ColdStorage(str(tmp_path)), sparsifier=None)
    request = FakeRequest()
    callbacks.before_model(context, request)

    assert request.contents[-1].parts[0].function_call.name == "run_shell_command"


def test_guard_blocks_dangerous_tool():
    class Guard:
        def check(self, tool_name, args, context):
            return tool_name != "run_shell_command"

    class Tool:
        name = "run_shell_command"

    callbacks = ADKAgentCallbacks(guards=[Guard()])

    result = callbacks.before_tool(Tool(), {}, None)

    assert result == {"error": "Tool 'run_shell_command' blocked by security guard."}
