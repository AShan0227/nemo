"""Tests for SAC-inspired thermodynamic explorer."""

import random

from src.research.explorer import BoltzmannExplorer


def test_initial_state():
    explorer = BoltzmannExplorer()
    assert explorer.temperature == 1.0
    assert explorer.stats["known_states"] == 0


def test_uniform_selection_high_temp():
    """High temperature should give near-uniform selection."""
    random.seed(42)
    explorer = BoltzmannExplorer(initial_temperature=100.0)
    actions = ["tap", "scroll", "back"]

    counts = {a: 0 for a in actions}
    for _ in range(300):
        action, _ = explorer.select_action("state1", actions)
        counts[action] += 1

    # Should be roughly uniform
    for count in counts.values():
        assert count > 50  # each action selected at least ~16% of time


def test_greedy_selection_low_temp():
    """Low temperature should favor high-value actions."""
    random.seed(42)
    explorer = BoltzmannExplorer(initial_temperature=0.01, entropy_bonus=0.0)
    explorer.update_value("s1", "tap", reward=10.0)
    explorer.update_value("s1", "scroll", reward=1.0)

    tap_count = 0
    for _ in range(100):
        action, _ = explorer.select_action("s1", ["tap", "scroll"])
        if action == "tap":
            tap_count += 1

    assert tap_count > 90  # should almost always pick "tap"


def test_value_update():
    explorer = BoltzmannExplorer()
    explorer.update_value("s1", "tap", reward=5.0)
    values = explorer.get_action_values("s1")
    assert len(values) == 1
    assert values[0].action_name == "tap"
    assert values[0].q_value == 5.0


def test_annealing():
    explorer = BoltzmannExplorer(initial_temperature=1.0, annealing_rate=0.5)
    explorer.anneal()
    assert explorer.temperature == 0.5
    explorer.anneal()
    assert explorer.temperature == 0.25


def test_min_temperature():
    explorer = BoltzmannExplorer(initial_temperature=0.2, min_temperature=0.1, annealing_rate=0.1)
    for _ in range(100):
        explorer.anneal()
    assert explorer.temperature >= 0.1


def test_policy_entropy():
    explorer = BoltzmannExplorer(initial_temperature=1.0)
    # No values = uniform = max entropy
    entropy = explorer.compute_policy_entropy("unknown", ["a", "b", "c"])
    assert entropy > 1.0  # close to log2(3) ≈ 1.585
