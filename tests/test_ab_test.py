"""Tests for A/B prompt/provider evaluation utilities."""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.ab_test import ABTestCase, ABVariant, run_ab_test


class FakeABClient:
    async def decide_action_structured(
        self,
        system_prompt: str,
        screen_context: str,
        task: str,
        *,
        history: list[dict[str, Any]] | None = None,
        prompt_version: str | None = None,
        provider_override: str | None = None,
    ) -> dict[str, Any]:
        del screen_context, task, history
        if provider_override == "gpt":
            action = "tap"
            entropy = 0.2
        elif prompt_version == "baseline_v1":
            action = "wait"
            entropy = 0.7
        else:
            action = "tap"
            entropy = 0.3
        return {
            "action": action,
            "params": {"index": 0},
            "reasoning": system_prompt[:10],
            "_meta": {
                "provider": provider_override or "qwen",
                "entropy": entropy,
                "entropy_source": "logprobs",
            },
        }


@pytest.mark.asyncio
async def test_run_ab_test_reports_accuracy_and_entropy():
    client = FakeABClient()
    cases = [
        ABTestCase(
            name="settings",
            task="Open network",
            screen_context="[0] Network",
            expected_actions=["tap"],
        ),
        ABTestCase(
            name="chat",
            task="Send hi",
            screen_context="[0] Message",
            expected_actions=["tap", "type_text"],
        ),
    ]
    variants = [
        ABVariant(
            name="baseline",
            system_prompt="baseline",
            prompt_version="baseline_v1",
            provider=None,
        ),
        ABVariant(
            name="gpt-reflect",
            system_prompt="reflect",
            prompt_version="reflect_fewshot_v1",
            provider="gpt",
        ),
    ]

    reports = await run_ab_test(client, cases, variants)
    assert len(reports) == 2

    baseline = reports[0]
    assert baseline.total_cases == 2
    assert baseline.matched_cases == 0
    assert baseline.action_accuracy == 0.0
    assert baseline.avg_entropy == pytest.approx(0.7)

    improved = reports[1]
    assert improved.matched_cases == 2
    assert improved.action_accuracy == 1.0
    assert improved.format_error_rate == 0.0
    assert improved.avg_entropy == pytest.approx(0.2)
