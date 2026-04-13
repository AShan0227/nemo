"""Tests for agent action building, routing, and graph wiring."""

from src.agent.agent import PhoneAgent
from src.config.settings import AgentConfig
from src.device.actions import ActionType
from src.knowledge.graph import ScreenGraph
from src.llm.router import EntropyRouter, ReasoningDepth
from src.screen.parser import parse_ui_hierarchy

from tests.conftest import SETTINGS_XML, WECHAT_CHAT_XML


def _make_agent() -> PhoneAgent:
    config = AgentConfig(max_steps=5, action_delay_ms=0, fusion_enabled=False)
    return PhoneAgent(config)


# --- _build_action tests ---


def test_build_tap():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("tap", {"index": 0}, screen)
    assert action is not None
    assert action.type == ActionType.TAP
    assert "x" in action.params and "y" in action.params


def test_build_type_text():
    """type_text must produce TYPE_TEXT with x, y, and text."""
    agent = _make_agent()
    screen = parse_ui_hierarchy(WECHAT_CHAT_XML)
    action = agent._build_action("type_text", {"index": 0, "text": "hello"}, screen)
    assert action is not None
    assert action.type == ActionType.TYPE_TEXT
    assert action.params["text"] == "hello"
    assert "x" in action.params
    assert "y" in action.params


def test_build_scroll():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("scroll", {"direction": "up"}, screen)
    assert action is not None
    assert action.type == ActionType.SCROLL


def test_build_back():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("back", {}, screen)
    assert action.type == ActionType.BACK


def test_build_home():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("home", {}, screen)
    assert action.type == ActionType.HOME


def test_build_launch():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("launch", {"package": "com.tencent.mm"}, screen)
    assert action.type == ActionType.LAUNCH_APP
    assert action.params["package"] == "com.tencent.mm"


def test_build_wait():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("wait", {"ms": 2000}, screen)
    assert action.type == ActionType.WAIT
    assert action.params["ms"] == 2000


def test_build_done():
    """done should return None (handled in execute before builder)."""
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("done", {"summary": "task complete"}, screen)
    assert action is None


def test_build_unknown():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("unknown_action", {}, screen)
    assert action is None


def test_build_tap_invalid_index():
    agent = _make_agent()
    screen = parse_ui_hierarchy(SETTINGS_XML)
    action = agent._build_action("tap", {"index": 999}, screen)
    assert action is None


# --- Entropy routing tests ---


def test_router_entropy_computation():
    router = EntropyRouter(0.3, 0.7)
    entropy = router.compute_entropy([1.0])
    assert entropy == 0.0  # single option = no uncertainty

    entropy = router.compute_entropy([0.5, 0.5])
    assert abs(entropy - 1.0) < 0.01  # max entropy for 2 options


def test_router_system1_cache():
    router = EntropyRouter(0.3, 0.7)
    router.cache_action("screen_abc", '{"action":"tap"}')

    decision = router.route("screen_abc", [0.99, 0.01])
    assert decision.depth == ReasoningDepth.SYSTEM_1


def test_router_system2_unknown():
    router = EntropyRouter(0.3, 0.7)
    decision = router.route("unknown_screen", [0.25, 0.25, 0.25, 0.25])
    assert decision.depth == ReasoningDepth.SYSTEM_2


def test_route_decision_graph_proxy():
    """Agent routing uses graph visit counts as entropy proxy."""
    agent = _make_agent()
    # Unknown screen -> high entropy (System 1.5 or System 2)
    decision = agent._route_decision("unknown_hash")
    assert decision.depth != ReasoningDepth.SYSTEM_1  # should not be cached


# --- Knowledge graph wiring tests ---


def test_graph_hash():
    hash1 = ScreenGraph.compute_screen_hash("<xml>screen1</xml>")
    hash2 = ScreenGraph.compute_screen_hash("<xml>screen2</xml>")
    assert hash1 != hash2
    assert len(hash1) == 12


def test_graph_record_and_query():
    g = ScreenGraph()
    g.record_transition("s1", "s2", "tap", success=True)
    g.record_transition("s1", "s2", "tap", success=True)
    edges = g.get_neighbors("s1")
    assert len(edges) == 1
    assert edges[0].success_count == 2
    assert edges[0].pheromone > 1.0  # reinforced
