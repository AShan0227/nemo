"""Integration tests for the PhoneAgent execute loop with mocked dependencies."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch, PropertyMock

import pytest

from src.agent.agent import PhoneAgent, TaskStatus
from src.config.settings import AgentConfig

from tests.conftest import (
    SETTINGS_XML, WIFI_XML, WECHAT_CHAT_XML, PURCHASE_XML,
    make_llm_response,
)


def _make_agent(max_steps: int = 30, safety: bool = True) -> PhoneAgent:
    config = AgentConfig(
        max_steps=max_steps,
        action_delay_ms=0,
        safety_enabled=safety,
        fusion_enabled=False,
    )
    return PhoneAgent(config)


def _mock_adb(agent: PhoneAgent, xml_sequence: list[str]) -> None:
    """Replace ADB controller with mocks returning a sequence of XML screens."""
    idx = {"i": 0}

    async def mock_hierarchy():
        xml = xml_sequence[min(idx["i"], len(xml_sequence) - 1)]
        idx["i"] += 1
        return xml

    agent.adb = AsyncMock()
    agent.adb.get_ui_hierarchy = mock_hierarchy
    agent.adb.tap = AsyncMock()
    agent.adb.input_text = AsyncMock()
    agent.adb.swipe = AsyncMock()
    agent.adb.press_back = AsyncMock()
    agent.adb.press_home = AsyncMock()
    agent.adb.launch_app = AsyncMock()
    agent.adb.get_screen_size = AsyncMock(return_value=(1080, 1920))


def _mock_llm(agent: PhoneAgent, responses: list[str]) -> None:
    """Replace LLM client with mocks returning scripted responses."""
    idx = {"i": 0}

    async def mock_decide(*args, **kwargs):
        resp = responses[min(idx["i"], len(responses) - 1)]
        idx["i"] += 1
        return resp

    agent.llm = AsyncMock()
    agent.llm.decide_action = mock_decide


@pytest.mark.asyncio
async def test_simple_tap_task():
    """LLM says tap element 1, then done. Verify completion."""
    agent = _make_agent()
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("done", {"summary": "Opened Wi-Fi"}, "Task complete"),
    ])

    result = await agent.execute("Open Wi-Fi settings")
    assert result.status == TaskStatus.COMPLETED
    assert result.total_steps == 2
    assert result.steps[0].action == "tap"
    assert result.steps[1].action == "done"


@pytest.mark.asyncio
async def test_type_text_task():
    """Verify type_text taps element then types text."""
    agent = _make_agent(safety=False)  # chat screen triggers message_send_guard
    _mock_adb(agent, [WECHAT_CHAT_XML, WECHAT_CHAT_XML, WECHAT_CHAT_XML])
    _mock_llm(agent, [
        make_llm_response("type_text", {"index": 0, "text": "hello"}, "Type message"),
        make_llm_response("done", {"summary": "Typed message"}, "Done"),
    ])

    result = await agent.execute("Type hello in chat")
    assert result.status == TaskStatus.COMPLETED
    # Verify tap was called (for focus) then input_text
    agent.adb.tap.assert_called()
    agent.adb.input_text.assert_called_once_with("hello")


@pytest.mark.asyncio
async def test_multi_step_task():
    """3 actions then done."""
    agent = _make_agent(safety=False)
    # Provide enough screens for graph wiring (observe before + observe after each action)
    _mock_adb(agent, [SETTINGS_XML, SETTINGS_XML, SETTINGS_XML, WIFI_XML,
                       WIFI_XML, WIFI_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("launch", {"package": "com.android.settings"}, "Open settings"),
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("tap", {"index": 0}, "Tap toggle"),
        make_llm_response("done", {"summary": "Wi-Fi toggled"}, "Complete"),
    ])

    result = await agent.execute("Toggle WiFi")
    assert result.status == TaskStatus.COMPLETED
    assert result.total_steps == 4


@pytest.mark.asyncio
async def test_safety_block():
    """Action on purchase screen should be blocked."""
    agent = _make_agent()
    _mock_adb(agent, [PURCHASE_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 0}, "Tap pay button"),
    ])

    result = await agent.execute("Buy the item")
    assert result.status == TaskStatus.BLOCKED
    assert "financial_guard" in result.summary


@pytest.mark.asyncio
async def test_max_steps_exceeded():
    """Verify FAILED when max steps reached (safety disabled to avoid step_limit block)."""
    agent = _make_agent(max_steps=2, safety=False)
    _mock_adb(agent, [SETTINGS_XML] * 20)
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 0}, "Tap something"),
    ] * 20)

    result = await agent.execute("Do something endlessly")
    assert result.status == TaskStatus.FAILED
    assert "Max steps" in result.summary


@pytest.mark.asyncio
async def test_parse_error_recovery():
    """LLM returns garbage, agent should continue to next step."""
    agent = _make_agent(max_steps=3)
    _mock_adb(agent, [SETTINGS_XML] * 5)
    _mock_llm(agent, [
        "This is not JSON at all",
        make_llm_response("done", {"summary": "recovered"}, "OK"),
    ])

    result = await agent.execute("Test parse error")
    assert result.status == TaskStatus.COMPLETED
    assert result.steps[0].action == "error"
    assert result.steps[0].success is False
    assert result.steps[1].action == "done"


@pytest.mark.asyncio
async def test_graph_records_transitions():
    """Verify knowledge graph is updated after actions."""
    agent = _make_agent()
    # Need 2+ non-done actions so prev_screen_hash is set before transition recording
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML, WIFI_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("tap", {"index": 0}, "Tap toggle"),
        make_llm_response("done", {"summary": "Done"}, "Complete"),
    ])

    result = await agent.execute("Toggle Wi-Fi")
    assert result.status == TaskStatus.COMPLETED

    # Graph should have nodes (at least from visited screens)
    assert len(agent.graph._nodes) > 0
    # Step 2 has prev_screen_hash set, so transition should be recorded
    total_edges = sum(len(edges) for edges in agent.graph._edges.values())
    assert total_edges >= 1
