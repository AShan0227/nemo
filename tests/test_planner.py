"""Tests for task planner graph + LLM fallback planning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.agent.planner import TaskPlanner
from src.knowledge.graph import ScreenGraph


def test_plan_from_graph_returns_steps():
    graph = ScreenGraph()
    graph.record_transition("home", "settings", "tap", success=True)
    graph.record_transition("settings", "wifi", "tap", success=True)
    planner = TaskPlanner(graph)

    plan = planner.plan_from_graph("home", "wifi")
    assert plan is not None
    assert len(plan.steps) == 2
    assert plan.steps[0].action_type == "tap"


@pytest.mark.asyncio
async def test_plan_with_llm_parses_json_array():
    graph = ScreenGraph()
    planner = TaskPlanner(graph)
    payload = (
        '[{"step":1,"action_name":"launch","params":{"package":"com.android.settings"},'
        '"action":"open settings","expected_result":"settings opened"}]'
    )

    async def fake_chat(*args, **kwargs):
        return {"content": payload}

    llm = SimpleNamespace(chat=fake_chat)
    plan = await planner.plan_with_llm(
        llm,
        task="open wifi",
        current_app="launcher",
        screen_summary="home screen",
    )
    assert plan is not None
    assert plan.steps[0].action_type == "launch"
    assert plan.steps[0].params["package"] == "com.android.settings"


def test_plan_next_action_hint():
    graph = ScreenGraph()
    planner = TaskPlanner(graph)
    plan = planner.plan_from_graph("a", "a")
    assert plan is not None
    assert planner.next_action_hint(plan, 0) is None
