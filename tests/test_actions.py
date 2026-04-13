"""Tests for action primitives, serialization, and recording."""

import json
import tempfile

from src.device.actions import Action, ActionRecording, ActionTimer, ActionType


# ---- Factory methods ----

def test_tap_factory():
    a = Action.tap(100, 200, "click it")
    assert a.type == ActionType.TAP
    assert a.params == {"x": 100, "y": 200}
    assert a.description == "click it"


def test_long_press_factory():
    a = Action.long_press(50, 60, 2000)
    assert a.type == ActionType.LONG_PRESS
    assert a.params["duration_ms"] == 2000


def test_double_tap_factory():
    a = Action.double_tap(100, 200)
    assert a.type == ActionType.DOUBLE_TAP
    assert a.params == {"x": 100, "y": 200}


def test_pinch_zoom_factory():
    a = Action.pinch_zoom(540, 960, zoom_in=True, scale=0.3)
    assert a.type == ActionType.PINCH_ZOOM
    assert a.params["zoom_in"] is True
    assert a.params["scale"] == 0.3


def test_all_factories():
    actions = [
        Action.tap(1, 2),
        Action.long_press(3, 4),
        Action.double_tap(5, 6),
        Action.swipe(1, 2, 3, 4),
        Action.type_text("hello"),
        Action.back(),
        Action.home(),
        Action.launch_app("com.test"),
        Action.scroll("down", 2),
        Action.wait(500),
        Action.pinch_zoom(100, 200),
    ]
    assert len(actions) == 11
    types = {a.type for a in actions}
    assert ActionType.DOUBLE_TAP in types
    assert ActionType.PINCH_ZOOM in types


# ---- Serialization ----

def test_action_to_dict():
    a = Action.tap(100, 200, "test")
    d = a.to_dict()
    assert d["type"] == "tap"
    assert d["params"] == {"x": 100, "y": 200}
    assert d["description"] == "test"


def test_action_roundtrip():
    original = Action.long_press(50, 60, 1500)
    d = original.to_dict()
    restored = Action.from_dict(d)
    assert restored.type == original.type
    assert restored.params == original.params


def test_action_json_roundtrip():
    a = Action.pinch_zoom(540, 960, zoom_in=False)
    json_str = json.dumps(a.to_dict())
    restored = Action.from_dict(json.loads(json_str))
    assert restored.type == ActionType.PINCH_ZOOM
    assert restored.params["zoom_in"] is False


# ---- Recording ----

def test_recording_add():
    rec = ActionRecording(name="test")
    a1 = Action.tap(1, 2)
    a1.duration_ms = 50
    a2 = Action.tap(3, 4)
    a2.duration_ms = 30
    rec.add(a1)
    rec.add(a2)
    assert len(rec.actions) == 2
    assert rec.total_duration_ms == 80


def test_recording_json_roundtrip():
    rec = ActionRecording(name="my_flow")
    rec.add(Action.tap(10, 20))
    rec.add(Action.type_text("hello"))
    rec.add(Action.back())

    json_str = rec.to_json()
    restored = ActionRecording.from_json(json_str)
    assert restored.name == "my_flow"
    assert len(restored.actions) == 3
    assert restored.actions[1].type == ActionType.TYPE_TEXT


def test_recording_save_load(tmp_path):
    path = tmp_path / "recording.json"
    rec = ActionRecording(name="saved_flow")
    rec.add(Action.tap(100, 200))
    rec.add(Action.scroll("down"))
    rec.save(path)

    loaded = ActionRecording.load(path)
    assert loaded.name == "saved_flow"
    assert len(loaded.actions) == 2


# ---- ActionTimer ----

def test_action_timer():
    action = Action.tap(0, 0)
    assert action.duration_ms == 0.0
    with ActionTimer(action):
        total = sum(range(1000))  # some work
    assert action.duration_ms > 0.0


def test_timestamp_auto_set():
    a = Action.tap(0, 0)
    assert a.timestamp > 0
