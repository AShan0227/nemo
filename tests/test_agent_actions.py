"""Tests for agent action building, routing, and graph wiring."""

from pathlib import Path

from src.agent.agent import PhoneAgent, StepRecord
from src.config.settings import AgentConfig
from src.device.actions import ActionType
from src.knowledge.graph import ScreenGraph
from src.llm.router import (
    EntropyRouter,
    ReasoningDepth,
    RoutingBenchmarkCase,
    compare_routing_accuracy,
)
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


def test_route_decision_prefers_observed_entropy():
    """Agent routing should use observed real entropy when available."""
    agent = _make_agent()
    # Cold start: no entropy observed yet.
    decision = agent._route_decision("unknown_hash")
    assert decision.depth == ReasoningDepth.SYSTEM_2

    # After observing low entropy, this screen should route lighter.
    agent.router.observe_entropy("unknown_hash", 0.2)
    decision2 = agent._route_decision("unknown_hash")
    assert decision2.depth == ReasoningDepth.SYSTEM_1_5
    assert decision2.source in ("observed_entropy", "real_entropy")


def test_compare_routing_accuracy_real_entropy():
    cases = [
        RoutingBenchmarkCase(
            screen_hash="s1",
            expected_depth=ReasoningDepth.SYSTEM_2,
            action_probs=[0.7, 0.3],
            real_entropy=0.9,
            cached=False,
        ),
        RoutingBenchmarkCase(
            screen_hash="s2",
            expected_depth=ReasoningDepth.SYSTEM_1,
            action_probs=[0.95, 0.05],
            real_entropy=0.1,
            cached=True,
        ),
    ]
    report = compare_routing_accuracy(cases, threshold_low=0.3, threshold_high=0.7)
    assert report.sample_size == 2
    assert 0.0 <= report.heuristic_accuracy <= 1.0
    assert 0.0 <= report.real_entropy_accuracy <= 1.0


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


def test_agent_graph_persistence_helpers(tmp_path: Path):
    path = tmp_path / "graph.json"
    agent = PhoneAgent(AgentConfig(graph_persist_path=str(path), fusion_enabled=False))
    agent.graph.record_transition("a", "b", "tap", success=True)
    agent._save_graph_to_disk()

    agent2 = PhoneAgent(AgentConfig(graph_persist_path=str(path), fusion_enabled=False))
    agent2._load_graph_from_disk()
    assert len(agent2.graph.get_neighbors("a")) == 1


def test_agent_recent_context_window():
    agent = PhoneAgent(AgentConfig(context_window_steps=2, fusion_enabled=False))
    history = [
        StepRecord(1, "s1", "r1", "tap", {"index": 0}, True),
        StepRecord(2, "s2", "r2", "scroll", {"direction": "down"}, False, error="stuck"),
        StepRecord(3, "s3", "r3", "wait", {"ms": 1000}, True),
    ]
    payload = agent._build_recent_context(history)
    assert len(payload) == 2
    assert payload[0]["step"] == 2
    assert payload[1]["action"] == "wait"
