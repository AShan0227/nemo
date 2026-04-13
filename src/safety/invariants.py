"""Safety invariant layer — conservation-enforcing action filter.

Inspired by Noether's theorem: every symmetry implies a conservation law.
These are HARD constraints that can never be violated by the agent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from loguru import logger

from src.device.actions import Action


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


# --- Built-in invariants ---

FINANCIAL_KEYWORDS = {
    "pay",
    "purchase",
    "buy",
    "confirm order",
    "checkout",
    "transfer",
    "send money",
    "支付",
    "转账",
}
PERMISSION_KEYWORDS = {"allow", "grant", "permit", "authorize", "accept", "允许", "授权"}
DELETE_KEYWORDS = {"delete", "remove", "clear all", "erase", "卸载", "uninstall"}
MESSAGE_SEND_KEYWORDS = {
    "send",
    "send message",
    "sms",
    "message",
    "email",
    "mail",
    "发送",
    "短信",
    "邮件",
}
SETTINGS_PAGE_KEYWORDS = {"settings", "system settings", "developer options", "设置", "系统设置"}
SETTINGS_MUTATION_KEYWORDS = {"change", "modify", "enable", "disable", "开启", "关闭", "修改"}
APP_UNINSTALL_KEYWORDS = {"uninstall", "remove app", "卸载应用", "卸载", "删除应用"}


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _action_text(action: Action) -> str:
    return f"{action.description} {action.type.value} {action.params}".lower()


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


def _check_message_send(action: Action, ctx: dict) -> bool:
    """Block send actions unless both recipient and content were user-confirmed."""
    explicit_intent = bool(ctx.get("message_send_intent", False))
    inferred_intent = _contains_keyword(ctx.get("screen_text", ""), MESSAGE_SEND_KEYWORDS) and (
        _contains_keyword(_action_text(action), MESSAGE_SEND_KEYWORDS)
        or "send" in ctx.get("screen_text", "").lower()
    )
    if not explicit_intent and not inferred_intent:
        return True

    recipient_ok = bool(ctx.get("message_recipient_confirmed", False))
    content_ok = bool(ctx.get("message_content_confirmed", False))
    return recipient_ok and content_ok


def _check_system_settings_change(action: Action, ctx: dict) -> bool:
    """Block mutating system settings unless user confirmation is present."""
    if bool(ctx.get("system_settings_change_intent", False)):
        return bool(ctx.get("system_settings_change_confirmed", False))

    screen_text = ctx.get("screen_text", "")
    in_settings = _contains_keyword(screen_text, SETTINGS_PAGE_KEYWORDS)
    mutation_in_screen = _contains_keyword(screen_text, SETTINGS_MUTATION_KEYWORDS)
    mutation_in_action = _contains_keyword(_action_text(action), SETTINGS_MUTATION_KEYWORDS)
    mutation_detected = mutation_in_screen or mutation_in_action
    if not (in_settings and mutation_detected):
        return True
    return bool(ctx.get("system_settings_change_confirmed", False))


def _check_app_uninstall(action: Action, ctx: dict) -> bool:
    """Block app uninstall unless explicitly confirmed."""
    if bool(ctx.get("app_uninstall_intent", False)):
        return bool(ctx.get("app_uninstall_confirmed", False))

    uninstall_detected = _contains_keyword(ctx.get("screen_text", ""), APP_UNINSTALL_KEYWORDS) or (
        _contains_keyword(_action_text(action), APP_UNINSTALL_KEYWORDS)
    )
    if not uninstall_detected:
        return True
    return bool(ctx.get("app_uninstall_confirmed", False))


def _check_step_limit(action: Action, ctx: dict) -> bool:
    """Enforce maximum steps per task."""
    return ctx.get("step_count", 0) < ctx.get("max_steps", 30)


RULE_REGISTRY: dict[str, SafetyRule] = {
    "financial_guard": SafetyRule(
        "financial_guard",
        "No financial transactions without confirmation",
        _check_financial,
    ),
    "permission_guard": SafetyRule(
        "permission_guard",
        "No permission grants without confirmation",
        _check_permission,
    ),
    "deletion_guard": SafetyRule(
        "deletion_guard",
        "No destructive deletions without confirmation",
        _check_deletion,
    ),
    "message_send_guard": SafetyRule(
        "message_send_guard",
        "No SMS/email send without recipient and content confirmation",
        _check_message_send,
    ),
    "system_settings_guard": SafetyRule(
        "system_settings_guard",
        "No system settings changes without confirmation",
        _check_system_settings_change,
    ),
    "app_uninstall_guard": SafetyRule(
        "app_uninstall_guard",
        "No app uninstall without confirmation",
        _check_app_uninstall,
    ),
    "step_limit": SafetyRule(
        "step_limit",
        "Task must not exceed max steps",
        _check_step_limit,
    ),
}

DEFAULT_RULES: list[SafetyRule] = list(RULE_REGISTRY.values())


class SafetyLayer:
    """Filter all agent actions through safety invariants."""

    def __init__(
        self,
        rules: list[SafetyRule] | None = None,
        audit_log_path: str | Path | None = None,
    ) -> None:
        self._rules = rules or DEFAULT_RULES
        self._audit_log_path = Path(audit_log_path) if audit_log_path else None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> SafetyLayer:
        """Build a safety layer from YAML config."""
        config = _load_yaml_config(config_path)
        rules_config = config.get("rules", {})
        if rules_config is None:
            rules_config = {}
        if not isinstance(rules_config, dict):
            raise ValueError("`rules` must be a mapping in safety config")

        active_rules: list[SafetyRule] = []
        for rule_name, rule in RULE_REGISTRY.items():
            raw_rule_config = rules_config.get(rule_name, {})
            if raw_rule_config is None:
                raw_rule_config = {}
            if not isinstance(raw_rule_config, dict):
                raise ValueError(f"Rule config for `{rule_name}` must be a mapping")

            enabled = bool(raw_rule_config.get("enabled", True))
            if not enabled:
                continue

            description = str(raw_rule_config.get("description", rule.description))
            severity = str(raw_rule_config.get("severity", rule.severity))
            active_rules.append(
                SafetyRule(
                    name=rule.name,
                    description=description,
                    check=rule.check,
                    severity=severity,
                )
            )

        audit_log_path = config.get("audit_log_path", "./logs/safety_audit.log")
        if audit_log_path is not None and not isinstance(audit_log_path, str):
            raise ValueError("`audit_log_path` must be a string or null")

        return cls(rules=active_rules, audit_log_path=audit_log_path)

    def check(self, action: Action, screen_context: dict) -> SafetyResult:
        """Check if an action violates any safety invariant."""
        violations = []
        for rule in self._rules:
            if not rule.check(action, screen_context):
                violations.append(rule.name)
                logger.warning(f"Safety violation: {rule.name} — {rule.description}")

        if violations:
            message = f"Blocked by: {', '.join(violations)}. Needs user confirmation."
            self._write_audit_log(action, screen_context, violations, message)
            return SafetyResult(
                verdict=Verdict.BLOCK,
                original_action=action,
                violated_rules=violations,
                message=message,
            )
        return SafetyResult(verdict=Verdict.ALLOW, original_action=action)

    def _write_audit_log(
        self,
        action: Action,
        screen_context: dict,
        violated_rules: list[str],
        message: str,
    ) -> None:
        """Persist interception events to an append-only JSONL audit log."""
        if self._audit_log_path is None:
            return

        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017 (py39 compatibility)
            "verdict": Verdict.BLOCK.value,
            "violated_rules": violated_rules,
            "message": message,
            "action": {
                "type": action.type.value,
                "description": action.description,
                "params": action.params,
            },
            "context": {
                "step_count": screen_context.get("step_count"),
                "max_steps": screen_context.get("max_steps"),
            },
        }

        try:
            self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.error(f"Failed to write safety audit log {self._audit_log_path}: {exc}")


def _load_yaml_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open(encoding="utf-8") as f:
        raw = f.read()

    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError(
            "PyYAML is required for YAML safety config. Install with `pip install pyyaml`."
        ) from exc

    data = yaml.safe_load(raw)  # type: ignore[no-any-return]
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Safety config file must contain a top-level mapping")
    return data
