"""Tests for action serialization, recording, and timing metrics."""

from __future__ import annotations

from pathlib import Path

from src.device.actions import Action, ActionRecorder, ActionTimingTracker, ActionType


def test_action_roundtrip_dict():
    action = Action.pinch_zoom(100, 200, distance=180, zoom_in=False, duration_ms=500)
    payload = action.to_dict()
    restored = Action.from_dict(payload)

    assert restored.type == action.type
    assert restored.params == action.params
    assert restored.is_reversible == action.is_reversible


def test_action_recorder_save_load(tmp_path: Path):
    recorder = ActionRecorder()
    recorder.record(
        timestamp_ms=1_700_000_000_000,
        action=Action.tap(10, 20, "Tap login"),
        success=True,
        duration_ms=123.4,
    )
    recorder.record(
        timestamp_ms=1_700_000_000_200,
        action=Action.scroll("down", 2),
        success=False,
        error="timeout",
        duration_ms=300.0,
        attempts=2,
    )

    path = recorder.save(tmp_path / "records.json")
    loaded = ActionRecorder.load(path)

    assert len(loaded.records) == 2
    assert loaded.records[0].action.type == ActionType.TAP
    assert loaded.records[1].success is False
    assert loaded.records[1].error == "timeout"
    assert loaded.records[1].attempts == 2


def test_action_timing_tracker_snapshot():
    tracker = ActionTimingTracker()
    tracker.record(ActionType.TAP, 100.0, success=True)
    tracker.record(ActionType.TAP, 200.0, success=False)
    tracker.record(ActionType.SCROLL, 300.0, success=True)

    snapshot = tracker.snapshot()
    assert snapshot["tap"]["count"] == 2.0
    assert snapshot["tap"]["success_count"] == 1.0
    assert snapshot["tap"]["avg_duration_ms"] == 150.0
    assert snapshot["scroll"]["success_rate"] == 1.0
