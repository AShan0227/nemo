"""Tests for recording -> Knowledge Graph import."""

from src.device.actions import Action, ActionRecord, ActionRecorder
from src.knowledge.graph import ScreenGraph
from src.knowledge.recording_import import (
    import_recording_file,
    merge_recordings,
    recording_to_graph,
)


def _make_recorder(*action_types: str) -> ActionRecorder:
    """Helper to create a simple recorder with given action types."""
    recorder = ActionRecorder()
    for i, atype in enumerate(action_types):
        action = Action.tap(100, 200) if atype == "tap" else Action.scroll("down")
        recorder.record(
            timestamp_ms=1000 * i,
            action=action,
            success=True,
            duration_ms=50.0,
        )
    return recorder


def test_recording_to_graph_basic():
    recorder = _make_recorder("tap", "tap", "scroll")
    graph = recording_to_graph(recorder)
    assert len(graph._nodes) == 4  # 3 actions + 1 = 4 screen states
    total_edges = sum(len(e) for e in graph._edges.values())
    assert total_edges == 3


def test_recording_to_graph_with_screen_hashes():
    recorder = _make_recorder("tap", "scroll")
    hashes = ["home", "list", "detail"]
    graph = recording_to_graph(recorder, screen_hashes=hashes)
    assert "home" in graph._nodes
    assert "list" in graph._nodes
    assert "detail" in graph._nodes


def test_recording_to_graph_wrong_hash_count():
    """Mismatched hash count should fall back to synthetic hashes."""
    recorder = _make_recorder("tap")
    graph = recording_to_graph(recorder, screen_hashes=["only_one"])
    assert len(graph._nodes) == 2  # synthetic hashes used


def test_recording_to_graph_empty():
    recorder = ActionRecorder()
    graph = recording_to_graph(recorder)
    assert len(graph._nodes) == 0


def test_recording_to_graph_merge_into_existing():
    existing = ScreenGraph()
    existing.add_node(
        __import__("src.knowledge.graph", fromlist=["ScreenNode"]).ScreenNode(
            state_id="shared_screen", ui_hash="shared_screen"
        )
    )

    recorder = _make_recorder("tap")
    hashes = ["shared_screen", "new_screen"]
    graph = recording_to_graph(recorder, screen_hashes=hashes, graph=existing)
    assert "shared_screen" in graph._nodes
    assert "new_screen" in graph._nodes


def test_merge_recordings():
    rec1 = _make_recorder("tap", "tap")
    rec2 = _make_recorder("scroll")
    hashes1 = ["A", "B", "C"]
    hashes2 = ["B", "D"]  # shares "B" with rec1

    graph = merge_recordings(
        [rec1, rec2],
        screen_hashes_list=[hashes1, hashes2],
    )
    assert "A" in graph._nodes
    assert "B" in graph._nodes
    assert "C" in graph._nodes
    assert "D" in graph._nodes


def test_import_recording_file(tmp_path):
    recorder = _make_recorder("tap", "scroll", "tap")
    path = recorder.save(tmp_path / "trace.json")
    graph = import_recording_file(path)
    assert len(graph._nodes) == 4


def test_screen_hash_in_action_record():
    record = ActionRecord(
        timestamp_ms=1000,
        action=Action.tap(10, 20),
        success=True,
        screen_hash="abc123",
    )
    d = record.to_dict()
    assert d["screen_hash"] == "abc123"
    restored = ActionRecord.from_dict(d)
    assert restored.screen_hash == "abc123"


def test_screen_hash_empty_by_default():
    record = ActionRecord(
        timestamp_ms=1000,
        action=Action.tap(10, 20),
        success=True,
    )
    d = record.to_dict()
    assert "screen_hash" not in d  # empty hash omitted
