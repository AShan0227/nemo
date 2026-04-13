"""Integration tests for research mechanisms wired into PhoneAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.agent.agent import PhoneAgent, TaskStatus
from src.config.settings import AgentConfig

from tests.conftest import SETTINGS_XML, WIFI_XML, make_llm_response


def _make_agent(**kwargs) -> PhoneAgent:
    defaults = dict(
        max_steps=30, action_delay_ms=0, safety_enabled=False,
        fusion_enabled=False, verify_enabled=False,
        homeostasis_enabled=True, inertia_enabled=True,
        phase_detector_enabled=True, immune_enabled=False,
        explorer_enabled=False,
    )
    defaults.update(kwargs)
    return PhoneAgent(AgentConfig(**defaults))


def _mock_adb(agent: PhoneAgent, xml_sequence: list[str]) -> None:
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
    idx = {"i": 0}

    async def mock_decide(*args, **kwargs):
        resp = responses[min(idx["i"], len(responses) - 1)]
        idx["i"] += 1
        return resp

    agent.llm = AsyncMock()
    agent.llm.decide_action = mock_decide


@pytest.mark.asyncio
async def test_homeostasis_updates_metrics():
    """Homeostasis regulator should receive metrics during execution."""
    agent = _make_agent(homeostasis_enabled=True)
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("done", {"summary": "Done"}, "Complete"),
    ])

    result = await agent.execute("Open Wi-Fi")
    assert result.status == TaskStatus.COMPLETED
    # Homeostasis should have received at least one update
    sr = agent.homeostasis.variables["success_rate"]
    assert len(sr.history) > 0


@pytest.mark.asyncio
async def test_inertia_initialized():
    """Inertia controller should be active when enabled."""
    agent = _make_agent(inertia_enabled=True)
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 0}, "Tap settings"),
        make_llm_response("done", {"summary": "Done"}),
    ])

    result = await agent.execute("Open settings")
    assert result.status == TaskStatus.COMPLETED
    assert agent.inertia is not None


@pytest.mark.asyncio
async def test_phase_detector_records_snapshot():
    """Phase detector should receive a snapshot after task completion."""
    agent = _make_agent(phase_detector_enabled=True)
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("done", {"summary": "Done"}),
    ])

    await agent.execute("Open Wi-Fi")
    # Phase detector should have at least 1 data point in success_rate history
    sr_history = agent.phase_detector._history["success_rate"]
    assert len(sr_history) >= 1


@pytest.mark.asyncio
async def test_explorer_updates_q_values():
    """Boltzmann explorer should update Q-values when enabled."""
    agent = _make_agent(explorer_enabled=True)
    _mock_adb(agent, [SETTINGS_XML, WIFI_XML, WIFI_XML, WIFI_XML, WIFI_XML])
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 1}, "Tap Wi-Fi"),
        make_llm_response("tap", {"index": 0}, "Tap toggle"),
        make_llm_response("done", {"summary": "Done"}),
    ])

    await agent.execute("Toggle Wi-Fi")
    assert agent.explorer.stats["known_states"] > 0
    assert agent.explorer.stats["total_actions"] > 0


@pytest.mark.asyncio
async def test_immune_training_phase():
    """Immune system should collect self samples when enabled but not trained."""
    agent = _make_agent(immune_enabled=True)
    _mock_adb(agent, [SETTINGS_XML] * 6)
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 0}, "Tap"),
        make_llm_response("done", {"summary": "Done"}),
    ])

    await agent.execute("Test task")
    # Should have collected self samples (but not enough to train with default 20)
    assert len(agent.immune._self_set) > 0


@pytest.mark.asyncio
async def test_all_mechanisms_together():
    """All research mechanisms enabled simultaneously should not crash."""
    agent = _make_agent(
        homeostasis_enabled=True, inertia_enabled=True,
        phase_detector_enabled=True, explorer_enabled=True,
        immune_enabled=True,
    )
    _mock_adb(agent, [SETTINGS_XML] * 10 + [WIFI_XML] * 10)
    _mock_llm(agent, [
        make_llm_response("tap", {"index": 0}, "Step 1"),
        make_llm_response("tap", {"index": 1}, "Step 2"),
        make_llm_response("done", {"summary": "All done"}),
    ])

    result = await agent.execute("Combined test")
    assert result.status == TaskStatus.COMPLETED
