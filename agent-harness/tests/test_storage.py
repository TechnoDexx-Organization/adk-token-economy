"""Ледник: полный журнал и точная обратимость через манифест."""
import json

from harness.modules.storage import ColdStorage


def read_entries(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_log_keeps_both_roles_with_identity(tmp_path):
    storage = ColdStorage(str(tmp_path))
    storage.log("s1", "user", "привет", invocation_id="inv1", event_id="e1", kind="user_message")
    storage.log("s1", "model", "здравствуй", invocation_id="inv1", event_id="e2", kind="model_message")

    entries = read_entries(tmp_path / "cold_storage.jsonl")

    assert [entry["role"] for entry in entries] == ["user", "model"]
    assert all(entry["sha256"] for entry in entries)
    assert entries[0]["invocation_id"] == "inv1"
    assert entries[0]["kind"] == "user_message"


def test_manifest_restores_original_exactly(tmp_path):
    storage = ColdStorage(str(tmp_path))
    original = "В общем, я устал"
    processed = "я устал"
    removed = [{"start": 0, "end": 9, "text": "В общем, "}]

    storage.log_manifest(
        "s1",
        event_id="e1",
        invocation_id="inv1",
        role="user",
        original=original,
        processed=processed,
        removed=removed,
        ops=["sparsify"],
    )

    manifest = read_entries(tmp_path / "cold_storage.jsonl")[0]
    restored = manifest["processed"]
    for span in sorted(manifest["removed"], key=lambda item: item["start"], reverse=True):
        restored = restored[: span["start"]] + span["text"] + restored[span["start"] :]

    assert restored == manifest["original"]
    assert manifest["kind"] == "manifest"
    assert manifest["ops"] == ["sparsify"]


def test_manifest_records_hashes(tmp_path):
    storage = ColdStorage(str(tmp_path))
    storage.log_manifest(
        "s1",
        event_id="e1",
        invocation_id="",
        role="user",
        original="оригинал",
        processed="ориг",
        removed=[],
        ops=["cap"],
    )

    manifest = read_entries(tmp_path / "cold_storage.jsonl")[0]

    assert len(manifest["original_sha256"]) == 64
    assert len(manifest["processed_sha256"]) == 64
    assert manifest["original_length"] == len("оригинал")
