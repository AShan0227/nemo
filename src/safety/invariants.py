"""Safety invariant layer — conservation-enforcing action filter.

Inspired by Noether's theorem: every symmetry implies a conservation law.
These are HARD constraints that can never be violated by the agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

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

FINANCIAL_KEYWORDS = {"pay", "purchase", "buy", "confirm order", "checkout", "transfer", "send money"}
PERMISSION_KEYWORDS = {"allow", "grant", "permit", "authorize", "accept"}
DELETE_KEYWORDS = {"delete", "remove", "clear all", "erase", "uninstall"}


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


DEFAULT_RULES: list[SafetyRule] = [
    SafetyRule("financial_guard", "No financial transactions without confirmation", _check_financial),
    SafetyRule("permission_guard", "No permission grants without confirmation", _check_permission),
    SafetyRule("deletion_guard", "No destructive deletions without confirmation", _check_deletion),
    SafetyRule("step_limit", "Task must not exceed max steps", _check_step_limit),
]


class SafetyLayer:
    """Filter all agent actions through safety invariants."""

    def __init__(self, rules: list[SafetyRule] | None = None) -> None:
        self._rules = rules or DEFAULT_RULES

    def check(self, action: Action, screen_context: dict) -> SafetyResult:
        """Check if an action violates any safety invariant."""
        violations = []
        for rule in self._rules:
            if not rule.check(action, screen_context):
                violations.append(rule.name)
                logger.warning(f"Safety violation: {rule.name} — {rule.description}")

        if violations:
            return SafetyResult(
                verdict=Verdict.BLOCK,
                original_action=action,
                violated_rules=violations,
                message=f"Blocked by: {', '.join(violations)}. Needs user confirmation.",
            )
        return SafetyResult(verdict=Verdict.ALLOW, original_action=action)
