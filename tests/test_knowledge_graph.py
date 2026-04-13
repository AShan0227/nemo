"""Tests for screen state knowledge graph."""

from src.knowledge.graph import ActionEdge, ScreenGraph, ScreenNode


def test_a_star_basic():
    g = ScreenGraph()
    g.add_node(ScreenNode("home"))
    g.add_node(ScreenNode("settings"))
    g.add_node(ScreenNode("wifi"))
    g.add_edge(ActionEdge("home", "settings", "tap", cost=1.0))
    g.add_edge(ActionEdge("settings", "wifi", "tap", cost=1.0))

    path = g.a_star("home", "wifi")
    assert path is not None
    assert len(path) == 2
    assert path[0].to_state == "settings"
    assert path[1].to_state == "wifi"


def test_no_path():
    g = ScreenGraph()
    g.add_node(ScreenNode("a"))
    g.add_node(ScreenNode("b"))
    assert g.a_star("a", "b") is None


def test_record_transition():
    g = ScreenGraph()
    g.record_transition("home", "settings", "tap", success=True)
    g.record_transition("home", "settings", "tap", success=True)
    g.record_transition("home", "settings", "tap", success=False)

    edges = g.get_neighbors("home")
    assert len(edges) == 1
    assert edges[0].success_count == 2
    assert edges[0].failure_count == 1


def test_pheromone_evaporation():
    g = ScreenGraph()
    g.record_transition("a", "b", "tap", success=True)
    initial_pheromone = g.get_neighbors("a")[0].pheromone
    g.evaporate_pheromones(0.1)
    assert g.get_neighbors("a")[0].pheromone < initial_pheromone
