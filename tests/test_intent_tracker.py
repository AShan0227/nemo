"""Tests for intent tracker (particle filter)."""

from src.research.intent_tracker import IntentTracker


def test_initial_state():
    tracker = IntentTracker(["open system settings", "open app settings"])
    assert not tracker.is_collapsed
    state = tracker.get_state()
    assert len(state.hypotheses) == 2
    assert state.entropy > 0.9  # maximum uncertainty


def test_collapse_with_evidence():
    tracker = IntentTracker(
        ["open system settings", "open app settings"],
        collapse_threshold=0.7,
    )
    # Provide evidence matching "system settings"
    for _ in range(5):
        tracker.update_evidence("Wi-Fi Bluetooth Display System settings panel")

    assert tracker.is_collapsed
    assert "system" in tracker.get_best_intent()


def test_no_collapse_with_ambiguous_evidence():
    tracker = IntentTracker(
        ["open system settings", "open app settings"],
        collapse_threshold=0.9,
    )
    # Generic text that doesn't help disambiguate
    tracker.update_evidence("Home screen apps widgets")
    assert not tracker.is_collapsed


def test_should_ask_user():
    tracker = IntentTracker(
        ["search for headphones", "search for music"],
        max_steps_before_ask=3,
        collapse_threshold=0.95,
    )
    # Give vague evidence that doesn't collapse
    for _ in range(5):
        tracker.update_evidence("Welcome to the store")

    assert tracker.should_ask_user


def test_single_hypothesis():
    tracker = IntentTracker(["open camera"])
    assert tracker.get_best_intent() == "open camera"


def test_entropy_decreases():
    tracker = IntentTracker(
        ["send message to John", "call John"],
        collapse_threshold=0.8,
    )
    state0 = tracker.get_state()

    tracker.update_evidence("Chat input field Send button message John")
    state1 = tracker.get_state()

    # Entropy should decrease with relevant evidence
    assert state1.entropy <= state0.entropy
