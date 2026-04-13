"""Tests for friction/inertia plan stability."""

from src.research.inertia import InertiaController


def test_no_plan():
    ctrl = InertiaController()
    assert not ctrl.has_plan
    assert ctrl.compute_inertia() == 0.0


def test_set_plan():
    ctrl = InertiaController()
    ctrl.set_plan(["tap", "type_text", "tap", "done"])
    assert ctrl.has_plan
    assert ctrl._commitment.next_planned_action == "tap"


def test_agree_with_plan():
    ctrl = InertiaController()
    ctrl.set_plan(["tap", "scroll", "done"])
    decision = ctrl.should_follow_plan("tap", "tap")
    assert decision.use_planned
    assert decision.reason == "LLM agrees with plan"


def test_inertia_blocks_override():
    ctrl = InertiaController(base_inertia=0.5)
    ctrl.set_plan(["tap", "scroll", "done"])
    # Low confidence new action should be blocked
    decision = ctrl.should_follow_plan("tap", "scroll", new_action_confidence=0.3)
    assert decision.use_planned


def test_high_confidence_overrides():
    ctrl = InertiaController(base_inertia=0.3)
    ctrl.set_plan(["tap", "scroll", "done"])
    # High confidence new action should override
    decision = ctrl.should_follow_plan("tap", "scroll", new_action_confidence=0.9)
    assert not decision.use_planned


def test_inertia_increases_with_progress():
    ctrl = InertiaController(base_inertia=0.2, progress_weight=0.5)
    ctrl.set_plan(["a", "b", "c", "d", "e"])

    inertia_start = ctrl.compute_inertia()
    ctrl.advance()
    ctrl.advance()
    ctrl.advance()
    inertia_mid = ctrl.compute_inertia()

    assert inertia_mid > inertia_start


def test_clear_plan():
    ctrl = InertiaController()
    ctrl.set_plan(["tap", "done"])
    ctrl.clear_plan()
    assert not ctrl.has_plan
