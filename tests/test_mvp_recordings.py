"""Tests for MVP recording templates — validate they load and import to graph."""

from pathlib import Path

from src.device.actions import ActionRecorder
from src.knowledge.recording_import import import_recording_file, recording_to_graph


RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "mvp_recordings"


def test_wechat_recording_loads():
    path = RECORDINGS_DIR / "wechat_message.json"
    recorder = ActionRecorder.load(path)
    assert len(recorder.records) == 7
    assert recorder.records[0].action.type.value == "launch_app"
    assert recorder.records[-1].screen_hash == "wechat_chat_sent"


def test_settings_recording_loads():
    path = RECORDINGS_DIR / "settings_wifi.json"
    recorder = ActionRecorder.load(path)
    assert len(recorder.records) == 3
    assert recorder.records[0].action.type.value == "home"


def test_taobao_recording_loads():
    path = RECORDINGS_DIR / "taobao_search.json"
    recorder = ActionRecorder.load(path)
    assert len(recorder.records) == 4
    assert recorder.records[-1].action.type.value == "tap"


def test_wechat_recording_to_graph():
    path = RECORDINGS_DIR / "wechat_message.json"
    recorder = ActionRecorder.load(path)
    hashes = [r.screen_hash for r in recorder.records]
    # Add final screen state
    hashes.append("wechat_done")
    graph = recording_to_graph(recorder, screen_hashes=hashes)
    assert len(graph._nodes) >= 7


def test_all_recordings_import_to_graph():
    """All three MVP recordings should merge into one graph."""
    from src.knowledge.recording_import import merge_recordings

    recorders = []
    for name in ["wechat_message.json", "settings_wifi.json", "taobao_search.json"]:
        recorders.append(ActionRecorder.load(RECORDINGS_DIR / name))

    graph = merge_recordings(recorders)
    assert len(graph._nodes) > 0
    total_edges = sum(len(e) for e in graph._edges.values())
    # Synthetic hashes may collide across recordings (screen_0, screen_1, etc.)
    # so total edges <= sum of all actions
    assert total_edges >= 10  # at least most actions represented


def test_recording_screen_hashes_present():
    """All MVP recordings should have screen_hash on every record."""
    for name in ["wechat_message.json", "settings_wifi.json", "taobao_search.json"]:
        recorder = ActionRecorder.load(RECORDINGS_DIR / name)
        for record in recorder.records:
            assert record.screen_hash, f"{name}: record missing screen_hash"
