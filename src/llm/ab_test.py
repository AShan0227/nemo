"""A/B evaluation helpers for prompt/provider decision quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ABTestCase:
    """One evaluation sample."""

    name: str
    task: str
    screen_context: str
    expected_actions: list[str]
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ABVariant:
    """One variant under test (prompt + provider)."""

    name: str
    system_prompt: str
    prompt_version: str
    provider: str | None = None


@dataclass
class ABVariantReport:
    variant_name: str
    provider: str | None
    prompt_version: str
    total_cases: int
    matched_cases: int
    format_errors: int
    action_accuracy: float
    format_error_rate: float
    avg_entropy: float | None
    details: list[dict[str, Any]] = field(default_factory=list)


async def run_ab_test(
    llm_client: Any,
    cases: list[ABTestCase],
    variants: list[ABVariant],
) -> list[ABVariantReport]:
    """Run A/B evaluation across prompt/provider variants.

    `llm_client` must expose `decide_action_structured(...)`.
    """
    reports: list[ABVariantReport] = []

    for variant in variants:
        matched = 0
        format_errors = 0
        entropy_samples: list[float] = []
        details: list[dict[str, Any]] = []

        for case in cases:
            decision = await llm_client.decide_action_structured(
                variant.system_prompt,
                case.screen_context,
                case.task,
                history=case.history,
                prompt_version=variant.prompt_version,
                provider_override=variant.provider,
            )

            action = str(decision.get("action", "")).strip()
            params = decision.get("params", {})
            meta = decision.get("_meta", {})
            if not action:
                format_errors += 1

            expected = set(case.expected_actions)
            ok = bool(action and action in expected)
            if ok:
                matched += 1

            entropy = None
            if isinstance(meta, dict):
                raw_entropy = meta.get("entropy")
                if isinstance(raw_entropy, (int, float)):
                    entropy = float(raw_entropy)
                    entropy_samples.append(entropy)

            details.append(
                {
                    "case": case.name,
                    "expected_actions": sorted(expected),
                    "action": action,
                    "params": params,
                    "matched": ok,
                    "entropy": entropy,
                    "provider": meta.get("provider") if isinstance(meta, dict) else None,
                }
            )

        total = len(cases)
        reports.append(
            ABVariantReport(
                variant_name=variant.name,
                provider=variant.provider,
                prompt_version=variant.prompt_version,
                total_cases=total,
                matched_cases=matched,
                format_errors=format_errors,
                action_accuracy=(matched / total) if total else 0.0,
                format_error_rate=(format_errors / total) if total else 0.0,
                avg_entropy=(
                    (sum(entropy_samples) / len(entropy_samples))
                    if entropy_samples
                    else None
                ),
                details=details,
            )
        )

    return reports
