"""Tests for phase transition detector."""

import time

from src.research.phase_detector import PerformanceSnapshot, PhaseDetector


def _make_snapshot(success_rate: float, avg_steps: float = 10.0, t: float = 0.0) -> PerformanceSnapshot:
    return PerformanceSnapshot(
        timestamp=t,
        success_rate=success_rate,
        avg_steps=avg_steps,
        graph_nodes=10,
        graph_edges=15,
        task_count=1,
    )


def test_no_transition_with_little_data():
    """Detector should not fire with insufficient history."""
    detector = PhaseDetector(window_size=5, baseline_size=20)
    for i in range(4):
        result = detector.record(_make_snapshot(0.5, t=float(i)))
        assert result is None


def test_detect_success_rate_jump():
    """Sudden jump in success rate should be detected."""
    detector = PhaseDetector(window_size=5, baseline_size=50, sensitivity=1.5)

    # 30 snapshots at ~0.5 success rate (baseline)
    for i in range(30):
        detector.record(_make_snapshot(0.5, t=float(i)))

    # Sudden jump to 0.95
    transition = None
    for i in range(10):
        t = detector.record(_make_snapshot(0.95, t=float(30 + i)))
        if t is not None:
            transition = t

    assert transition is not None
    assert transition.metric_name == "success_rate"
    assert transition.new_value > transition.old_value


def test_no_transition_stable():
    """Stable metrics should not trigger detection."""
    detector = PhaseDetector(window_size=5, baseline_size=50, sensitivity=2.0)

    for i in range(50):
        result = detector.record(_make_snapshot(0.7, t=float(i)))

    # All at same level — no transition
    assert len(detector.transitions) == 0


def test_detect_steps_decrease():
    """Sudden decrease in avg_steps (improvement) should be detected."""
    detector = PhaseDetector(window_size=5, baseline_size=50, sensitivity=1.5)

    # Baseline: 15 steps average
    for i in range(30):
        detector.record(_make_snapshot(0.7, avg_steps=15.0, t=float(i)))

    # Sudden drop to 5 steps
    transition = None
    for i in range(10):
        t = detector.record(_make_snapshot(0.7, avg_steps=5.0, t=float(30 + i)))
        if t is not None:
            transition = t

    assert transition is not None
    assert transition.metric_name == "avg_steps"
    assert transition.new_value < transition.old_value
