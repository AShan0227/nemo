"""Tests for circadian rhythm modeling."""

import time

from src.research.circadian import CircadianModel


def test_empty_model():
    model = CircadianModel()
    assert model.coverage["total_activities"] == 0
    assert model.predict_apps() == []


def test_record_and_predict():
    model = CircadianModel()
    # Record morning WeChat usage
    morning_9am = 1713000000.0  # arbitrary timestamp
    for _ in range(5):
        model.record_activity("com.tencent.mm", "tap", timestamp=morning_9am)
    model.record_activity("com.android.settings", "tap", timestamp=morning_9am)

    predicted = model.predict_apps(timestamp=morning_9am)
    assert "com.tencent.mm" in predicted


def test_behavior_modifier_night():
    model = CircadianModel()
    # 3 AM — should be slow/quiet
    import datetime
    night = datetime.datetime(2026, 4, 13, 3, 0, 0).timestamp()
    mods = model.get_behavior_modifier(timestamp=night)
    assert mods["action_delay_multiplier"] > 1.0
    assert mods["proactive_suggestions"] == 0.0


def test_behavior_modifier_morning():
    model = CircadianModel()
    import datetime
    morning = datetime.datetime(2026, 4, 13, 8, 0, 0).timestamp()
    mods = model.get_behavior_modifier(timestamp=morning)
    assert mods["action_delay_multiplier"] < 1.0
    assert mods["proactive_suggestions"] > 0.5


def test_duration_tracking():
    model = CircadianModel()
    ts = 1713000000.0
    model.record_activity("com.app", "tap", duration_ms=1000, timestamp=ts)
    model.record_activity("com.app", "tap", duration_ms=3000, timestamp=ts)

    import datetime
    dt = datetime.datetime.fromtimestamp(ts)
    slot = model._get_slot(dt.hour, dt.weekday())
    # EMA should be between 1000 and 3000
    assert 1000 < slot.avg_task_duration_ms < 3000


def test_coverage():
    model = CircadianModel()
    model.record_activity("com.app", "tap", timestamp=1713000000.0)
    assert model.coverage["total_slots_observed"] == 1
    assert model.coverage["total_activities"] == 1
