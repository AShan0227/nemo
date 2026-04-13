"""Safety invariant layer — conservation-enforcing action filter.

Inspired by Noether's theorem: every symmetry implies a conservation law.
These are HARD constraints that can never be violated by the agent.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from src.device.actions import Action, ActionType


class Verdict(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


@dataclass
class SafetyRule:
    """A single safety invariant."""

    name: str
    description: str
    check: Callable[[Action, dict], bool]  # (action, screen_context) -> is_safe
    severity: str = "critical"  # critical | warning


@dataclass
class SafetyResult:
    verdict: Verdict
    original_action: Action
    safe_action: Action | None = None
    violated_rules: list[str] | None = None
    message: str = ""


# --- Keyword sets ---

FINANCIAL_KEYWORDS = {
    "pay", "purchase", "buy", "confirm order", "checkout",
    "transfer", "send money", "付款", "支付", "确认订单", "转账",
}
PERMISSION_KEYWORDS = {
    "allow", "grant", "permit", "authorize", "accept",
    "允许", "授权", "同意",
}
DELETE_KEYWORDS = {
    "delete", "remove", "clear all", "erase", "uninstall",
    "删除", "移除", "清除", "卸载",
}
MESSAGE_KEYWORDS = {
    "send", "发送", "send message", "send sms", "发短信", "发邮件",
    "send email", "forward", "转发", "reply all", "群发",
}
SYSTEM_SETTINGS_KEYWORDS = {
    "factory reset", "恢复出厂", "format", "格式化",
    "developer options", "开发者选项", "root", "bootloader",
    "wipe", "擦除",
}
UNINSTALL_KEYWORDS = {
    "uninstall", "卸载", "remove app", "删除应用",
}


# --- Built-in checks ---

def _check_financial(action: Action, ctx: dict) -> bool:
    """Block financial actions without explicit user confirmation."""
    screen_text = ctx.get("screen_text", "").lower()
    return not any(kw in screen_text for kw in FINANCIAL_KEYWORDS)


def _check_permission(action: Action, ctx: dict) -> bool:
    """Block granting permissions."""
    screen_text = ctx.get("screen_text", "").lower()
    return not any(kw in screen_text for kw in PERMISSION_KEYWORDS)


def _check_deletion(action: Action, ctx: dict) -> bool:
    """Block destructive deletions."""
    screen_text = ctx.get("screen_text", "").lower()
    return not any(kw in screen_text for kw in DELETE_KEYWORDS)


def _check_step_limit(action: Action, ctx: dict) -> bool:
    """Enforce maximum steps per task."""
    return ctx.get("step_count", 0) < ctx.get("max_steps", 30)


def _check_message_send(action: Action, ctx: dict) -> bool:
    """Block SMS/email/message sending without confirmation.

    Intercepts when the screen shows send-related keywords and
    the action targets a send button.
    """
    screen_text = ctx.get("screen_text", "").lower()
    if not any(kw in screen_text for kw in MESSAGE_KEYWORDS):
        return True
    # Only block tap/click actions on send-like buttons
    if action.type in (ActionType.TAP, ActionType.LONG_PRESS):
        action_desc = action.description.lower()
        return not any(kw in action_desc for kw in {"send", "发送", "submit", "提交"})
    return True


def _check_system_settings(action: Action, ctx: dict) -> bool:
    """Block dangerous system settings modifications."""
    screen_text = ctx.get("screen_text", "").lower()
    return not any(kw in screen_text for kw in SYSTEM_SETTINGS_KEYWORDS)


def _check_uninstall(action: Action, ctx: dict) -> bool:
    """Block app uninstallation."""
    screen_text = ctx.get("screen_text", "").lower()
    return not any(kw in screen_text for kw in UNINSTALL_KEYWORDS)


DEFAULT_RULES: list[SafetyRule] = [
    SafetyRule("financial_guard", "No financial transactions without confirmation", _check_financial),
    SafetyRule("permission_guard", "No permission grants without confirmation", _check_permission),
    SafetyRule("deletion_guard", "No destructive deletions without confirmation", _check_deletion),
    SafetyRule("step_limit", "Task must not exceed max steps", _check_step_limit),
    SafetyRule("message_guard", "No SMS/email sending without confirmation", _check_message_send),
    SafetyRule("system_settings_guard", "No dangerous system settings changes", _check_system_settings),
    SafetyRule("uninstall_guard", "No app uninstallation without confirmation", _check_uninstall),
]


# --- YAML-configurable safety rules ---

def load_rules_from_yaml(path: str | Path) -> list[SafetyRule]:
    """Load safety rules from a YAML config file.

    YAML format:
    ```yaml
    rules:
      - name: custom_block
        description: Block custom keywords
        keywords:
          - dangerous_word
          - another_bad_word
        severity: critical
        action_types: [tap, long_press]  # optional, defaults to all
    ```
    """
    import yaml

    content = Path(path).read_text()
    data = yaml.safe_load(content)
    rules: list[SafetyRule] = []

    for rule_def in data.get("rules", []):
        keywords = set(rule_def.get("keywords", []))
        allowed_types = rule_def.get("action_types")
        if allowed_types:
            allowed_types = {ActionType(t) for t in allowed_types}

        def make_check(kws: set[str], types: set[ActionType] | None):
            def check(action: Action, ctx: dict) -> bool:
                if types and action.type not in types:
                    return True
                screen_text = ctx.get("screen_text", "").lower()
                return not any(kw in screen_text for kw in kws)
            return check

        rules.append(
            SafetyRule(
                name=rule_def["name"],
                description=rule_def.get("description", ""),
                check=make_check(keywords, allowed_types),
                severity=rule_def.get("severity", "critical"),
            )
        )

    logger.info(f"Loaded {len(rules)} custom safety rules from {path}")
    return rules


# --- Audit Log ---

@dataclass
class AuditEntry:
    """A single safety audit log entry."""

    timestamp: float
    action_type: str
    verdict: str
    violated_rules: list[str]
    screen_context_summary: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "action_type": self.action_type,
            "verdict": self.verdict,
            "violated_rules": self.violated_rules,
            "screen_context_summary": self.screen_context_summary,
            "message": self.message,
        }


class SafetyAuditLog:
    """Persistent audit log for all safety check events."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._entries: list[AuditEntry] = []
        self._log_path = Path(log_path) if log_path else None

    def record(self, result: SafetyResult, screen_context: dict) -> None:
        """Record a safety check result."""
        entry = AuditEntry(
            timestamp=time.time(),
            action_type=result.original_action.type.value,
            verdict=result.verdict.value,
            violated_rules=result.violated_rules or [],
            screen_context_summary=screen_context.get("screen_text", "")[:200],
            message=result.message,
        )
        self._entries.append(entry)

        if result.verdict == Verdict.BLOCK:
            logger.warning(f"[AUDIT] BLOCKED: {entry.violated_rules} — {entry.message}")
        else:
            logger.debug(f"[AUDIT] ALLOWED: {result.original_action.type.value}")

        if self._log_path:
            self._flush()

    def _flush(self) -> None:
        """Append latest entries to disk."""
        if not self._log_path:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a") as f:
            entry = self._entries[-1]
            f.write(json.dumps(entry.to_dict()) + "\n")

    @property
    def entries(self) -> list[AuditEntry]:
        return list(self._entries)

    @property
    def block_count(self) -> int:
        return sum(1 for e in self._entries if e.verdict == "block")


# --- SafetyLayer ---

class SafetyLayer:
    """Filter all agent actions through safety invariants."""

    def __init__(
        self,
        rules: list[SafetyRule] | None = None,
        audit_log: SafetyAuditLog | None = None,
        custom_rules_path: str | Path | None = None,
    ) -> None:
        self._rules = list(rules or DEFAULT_RULES)
        self._audit_log = audit_log or SafetyAuditLog()

        # Load additional YAML rules if provided
        if custom_rules_path and Path(custom_rules_path).exists():
            self._rules.extend(load_rules_from_yaml(custom_rules_path))

    @property
    def audit_log(self) -> SafetyAuditLog:
        return self._audit_log

    def add_rule(self, rule: SafetyRule) -> None:
        """Dynamically add a safety rule."""
        self._rules.append(rule)

    def check(self, action: Action, screen_context: dict) -> SafetyResult:
        """Check if an action violates any safety invariant."""
        violations = []
        for rule in self._rules:
            if not rule.check(action, screen_context):
                violations.append(rule.name)
                logger.warning(f"Safety violation: {rule.name} — {rule.description}")

        if violations:
            result = SafetyResult(
                verdict=Verdict.BLOCK,
                original_action=action,
                violated_rules=violations,
                message=f"Blocked by: {', '.join(violations)}. Needs user confirmation.",
            )
        else:
            result = SafetyResult(verdict=Verdict.ALLOW, original_action=action)

        self._audit_log.record(result, screen_context)
        return result
