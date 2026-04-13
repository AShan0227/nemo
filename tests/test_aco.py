"""Tests for ACO transition probability selection."""

import random

from src.knowledge.graph import ActionEdge, ScreenGraph, ScreenNode


def test_aco_single_edge():
    """With one edge, always returns it."""
    g = ScreenGraph()
    g.record_transition("a", "b", "tap", success=True)
    edge = g.select_action_aco("a")
    assert edge is not None
    assert edge.to_state == "b"


def test_aco_no_edges():
    """No edges returns None."""
    g = ScreenGraph()
    g.add_node(ScreenNode("lonely"))
    assert g.select_action_aco("lonely") is None


def test_aco_exploitation():
    """With q0=1.0 (full exploitation), always picks highest score."""
    g = ScreenGraph()
    # Edge to "b" has high pheromone (many successes)
    for _ in range(10):
        g.record_transition("a", "b", "tap", success=True)
    # Edge to "c" has low pheromone (few successes)
    g.record_transition("a", "c", "tap", success=True)

    # Full exploitation should always pick "b"
    for _ in range(20):
        edge = g.select_action_aco("a", q0=1.0)
        assert edge.to_state == "b"


def test_aco_exploration():
    """With q0=0.0 (full exploration), both edges get selected over many trials."""
    random.seed(42)
    g = ScreenGraph()
    for _ in range(5):
        g.record_transition("a", "b", "tap", success=True)
    g.record_transition("a", "c", "tap", success=True)

    selected = set()
    for _ in range(100):
        edge = g.select_action_aco("a", q0=0.0)
        selected.add(edge.to_state)
    assert "b" in selected
    assert "c" in selected  # exploration should discover both


def test_aco_pheromone_affects_selection():
    """Higher pheromone edges get selected more often."""
    random.seed(42)
    g = ScreenGraph()
    # Make "b" much more attractive
    for _ in range(20):
        g.record_transition("a", "b", "tap", success=True)
    for _ in range(20):
        g.record_transition("a", "c", "tap", success=False)
    g.record_transition("a", "c", "tap", success=True)

    b_count = 0
    trials = 200
    for _ in range(trials):
        edge = g.select_action_aco("a", q0=0.5)
        if edge.to_state == "b":
            b_count += 1
    # "b" should be selected much more often than "c"
    assert b_count > trials * 0.6


def test_pheromone_min_bound():
    """Evaporation should not go below minimum."""
    g = ScreenGraph()
    g.record_transition("a", "b", "tap", success=False)
    # Evaporate many times
    for _ in range(100):
        g.evaporate_pheromones(0.5)
    edge = g.get_neighbors("a")[0]
    assert edge.pheromone >= 0.1
