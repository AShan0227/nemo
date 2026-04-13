"""Tests for safety invariant layer."""

import json
import tempfile
from pathlib import Path

from src.device.actions import Action
from src.safety.invariants import (
    SafetyAuditLog,
    SafetyLayer,
    SafetyRule,
    Verdict,
    load_rules_from_yaml,
)


def _ctx(text: str = "Home screen", step: int = 1, max_steps: int = 30) -> dict:
    return {"screen_text": text, "step_count": step, "max_steps": max_steps}


# ---- Existing tests (backward compatible) ----

def test_allow_normal_tap():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap button")
    result = layer.check(action, _ctx())
    assert result.verdict == Verdict.ALLOW


def test_block_financial():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "Tap button")
    result = layer.check(action, _ctx("Confirm purchase $99"))
    assert result.verdict == Verdict.BLOCK
    assert "financial_guard" in result.violated_rules


def test_block_step_limit():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("Normal screen", step=31))
    assert result.verdict == Verdict.BLOCK
    assert "step_limit" in result.violated_rules


def test_block_permission():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("Allow app to access location?"))
    assert result.verdict == Verdict.BLOCK


# ---- New rule tests ----

def test_block_message_send():
    layer = SafetyLayer()
    action = Action.tap(100, 200, "send button")
    result = layer.check(action, _ctx("发送短信给张三"))
    assert result.verdict == Verdict.BLOCK
    assert "message_guard" in result.violated_rules


def test_allow_message_without_send_button():
    """Reading a message screen (not tapping send) should be allowed."""
    layer = SafetyLayer()
    action = Action.tap(100, 200, "tap message preview")
    result = layer.check(action, _ctx("发送短信给张三"))
    assert result.verdict == Verdict.ALLOW


def test_block_system_settings():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("恢复出厂设置"))
    assert result.verdict == Verdict.BLOCK
    assert "system_settings_guard" in result.violated_rules


def test_block_uninstall():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("确认卸载此应用?"))
    assert result.verdict == Verdict.BLOCK
    assert "uninstall_guard" in result.violated_rules


def test_block_chinese_financial():
    layer = SafetyLayer()
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("确认支付 ¥99.00"))
    assert result.verdict == Verdict.BLOCK


# ---- Audit log tests ----

def test_audit_log_records():
    audit = SafetyAuditLog()
    layer = SafetyLayer(audit_log=audit)
    action = Action.tap(100, 200)
    layer.check(action, _ctx())
    layer.check(action, _ctx("Confirm purchase"))
    assert len(audit.entries) == 2
    assert audit.block_count == 1


def test_audit_log_persistence():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        log_path = f.name

    audit = SafetyAuditLog(log_path=log_path)
    layer = SafetyLayer(audit_log=audit)
    layer.check(Action.tap(100, 200), _ctx("delete everything"))

    content = Path(log_path).read_text()
    entry = json.loads(content.strip())
    assert entry["verdict"] == "block"
    assert "deletion_guard" in entry["violated_rules"]


# ---- YAML config tests ----

def test_load_yaml_rules():
    yaml_content = """
rules:
  - name: custom_danger
    description: Block dangerous keywords
    keywords:
      - very_dangerous
      - extremely_bad
    severity: critical
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    rules = load_rules_from_yaml(yaml_path)
    assert len(rules) == 1
    assert rules[0].name == "custom_danger"

    # Test the loaded rule works
    layer = SafetyLayer(rules=rules)
    action = Action.tap(100, 200)
    result = layer.check(action, _ctx("this is very_dangerous"))
    assert result.verdict == Verdict.BLOCK


def test_custom_rules_with_action_types():
    yaml_content = """
rules:
  - name: block_tap_on_bad
    description: Only block taps
    keywords:
      - bad_word
    action_types: [tap, long_press]
"""
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name

    rules = load_rules_from_yaml(yaml_path)
    layer = SafetyLayer(rules=rules)

    # Tap should be blocked
    tap = Action.tap(100, 200)
    assert layer.check(tap, _ctx("bad_word here")).verdict == Verdict.BLOCK

    # Scroll should NOT be blocked (not in action_types)
    scroll = Action.scroll("down")
    assert layer.check(scroll, _ctx("bad_word here")).verdict == Verdict.ALLOW


# ---- Dynamic rule addition ----

def test_add_rule_dynamically():
    layer = SafetyLayer(rules=[])
    action = Action.tap(100, 200)
    assert layer.check(action, _ctx("danger")).verdict == Verdict.ALLOW

    layer.add_rule(SafetyRule(
        name="dynamic",
        description="test",
        check=lambda a, c: "danger" not in c.get("screen_text", ""),
    ))
    assert layer.check(action, _ctx("danger")).verdict == Verdict.BLOCK
