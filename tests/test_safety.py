"""Tests for safety invariant layer."""

from src.device.actions import Action
from src.safety.invariants import SafetyLayer, Verdict


def test_allow_normal_tap():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap button")
    result = layer.check(action, {"screen_text": "Home screen", "step_count": 1, "max_steps": 30})
    assert result.verdict == Verdict.ALLOW


def test_block_financial():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap button")
    result = layer.check(action, {"screen_text": "Confirm purchase $99", "step_count": 1, "max_steps": 30})
    assert result.verdict == Verdict.BLOCK
    assert "financial_guard" in result.violated_rules


def test_block_step_limit():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, {"screen_text": "Normal screen", "step_count": 31, "max_steps": 30})
    assert result.verdict == Verdict.BLOCK
    assert "step_limit" in result.violated_rules


def test_block_permission():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, {"screen_text": "Allow app to access location?", "step_count": 1, "max_steps": 30})
    assert result.verdict == Verdict.BLOCK
