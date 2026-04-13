"""Tests for safety invariant layer."""

import json

import pytest

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
    result = layer.check(
        action,
        {"screen_text": "Confirm purchase $99", "step_count": 1, "max_steps": 30},
    )
    assert result.verdict == Verdict.BLOCK
    assert "financial_guard" in result.violated_rules


def test_block_step_limit():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(
        action,
        {"screen_text": "Normal screen", "step_count": 31, "max_steps": 30},
    )
    assert result.verdict == Verdict.BLOCK
    assert "step_limit" in result.violated_rules


def test_block_permission():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(
        action,
        {"screen_text": "Allow app to access location?", "step_count": 1, "max_steps": 30},
    )
    assert result.verdict == Verdict.BLOCK


def test_block_message_send_without_confirmations():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap send button")
    result = layer.check(
        action,
        {
            "screen_text": "SMS compose screen with Send",
            "step_count": 1,
            "max_steps": 30,
            "message_send_intent": True,
            "message_recipient_confirmed": False,
            "message_content_confirmed": False,
        },
    )
    assert result.verdict == Verdict.BLOCK
    assert "message_send_guard" in result.violated_rules


def test_allow_message_send_with_confirmations():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap send button")
    result = layer.check(
        action,
        {
            "screen_text": "SMS compose screen with Send",
            "step_count": 1,
            "max_steps": 30,
            "message_send_intent": True,
            "message_recipient_confirmed": True,
            "message_content_confirmed": True,
        },
    )
    assert result.verdict == Verdict.ALLOW


def test_block_system_settings_change_without_confirmation():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Enable WiFi")
    result = layer.check(
        action,
        {
            "screen_text": "System Settings WiFi",
            "step_count": 1,
            "max_steps": 30,
            "system_settings_change_intent": True,
            "system_settings_change_confirmed": False,
        },
    )
    assert result.verdict == Verdict.BLOCK
    assert "system_settings_guard" in result.violated_rules


def test_block_app_uninstall_without_confirmation():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Uninstall app")
    result = layer.check(
        action,
        {
            "screen_text": "Uninstall this app?",
            "step_count": 1,
            "max_steps": 30,
            "app_uninstall_intent": True,
            "app_uninstall_confirmed": False,
        },
    )
    assert result.verdict == Verdict.BLOCK
    assert "app_uninstall_guard" in result.violated_rules


def test_write_audit_log_for_blocked_action(tmp_path):
    log_path = tmp_path / "safety_audit.jsonl"
    layer = SafetyLayer(audit_log_path=log_path)
    action = Action.tap(100, 200, "Tap buy")

    result = layer.check(
        action,
        {"screen_text": "Confirm purchase $99", "step_count": 1, "max_steps": 30},
    )

    assert result.verdict == Verdict.BLOCK
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["verdict"] == "block"
    assert "financial_guard" in event["violated_rules"]


def test_load_rule_overrides_from_yaml(tmp_path):
    pytest.importorskip("yaml")
    config_path = tmp_path / "safety.yaml"
    config_path.write_text(
        "\n".join(
            [
                "rules:",
                "  financial_guard:",
                "    enabled: false",
                "audit_log_path: null",
            ]
        ),
        encoding="utf-8",
    )
    layer = SafetyLayer.from_yaml(config_path)
    action = Action.tap(100, 200)
    result = layer.check(
        action,
        {"screen_text": "Confirm purchase $99", "step_count": 1, "max_steps": 30},
    )
    assert result.verdict == Verdict.ALLOW
