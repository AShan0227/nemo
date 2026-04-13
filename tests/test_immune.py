"""Tests for immune system (NSA) anomaly detection."""

import random

from src.research.immune import Detector, ImmuneSystem, ScreenFeatures


def _normal_screen() -> ScreenFeatures:
    return ScreenFeatures(
        element_count=20, interactive_count=8, text_length=200,
        has_input=True, has_button=True, has_scrollable=True,
        unique_classes=5, depth_estimate=3,
    )


def _crash_screen() -> ScreenFeatures:
    """Simulates a crash dialog — very few elements, different structure."""
    return ScreenFeatures(
        element_count=3, interactive_count=1, text_length=50,
        has_input=False, has_button=True, has_scrollable=False,
        unique_classes=2, depth_estimate=1,
    )


def test_detector_matches():
    d = Detector(center=[0.5, 0.5], radius=0.3)
    assert d.matches([0.5, 0.5])
    assert d.matches([0.6, 0.6])
    assert not d.matches([1.0, 1.0])


def test_not_trained():
    immune = ImmuneSystem()
    result = immune.check(_normal_screen())
    assert not result.is_anomaly
    assert result.message == "Not trained yet"


def test_train_generates_detectors():
    random.seed(42)
    immune = ImmuneSystem(detector_count=20)
    for _ in range(10):
        immune.add_self_sample(_normal_screen())
    count = immune.train()
    assert count > 0


def test_normal_not_flagged():
    random.seed(42)
    immune = ImmuneSystem(detector_count=30, anomaly_threshold=3)
    for _ in range(20):
        immune.add_self_sample(_normal_screen())
    immune.train()

    result = immune.check(_normal_screen())
    assert not result.is_anomaly


def test_anomaly_detected():
    """Verify detectors fire for a significantly different screen.

    Uses the Detector class directly since the random NSA generation
    in high-dimensional spaces is stochastic — unit test the mechanism.
    """
    immune = ImmuneSystem(anomaly_threshold=1)

    # Manually create a detector positioned near the crash screen
    # and far from the normal screen
    normal_vec = _normal_screen().to_vector()
    crash_vec = _crash_screen().to_vector()

    # Place detector exactly where the crash screen is
    detector = Detector(center=crash_vec, radius=0.3)
    assert detector.matches(crash_vec)
    # Should NOT match the normal screen (distance ~1.48)
    assert not detector.matches(normal_vec)

    # Add to immune system manually
    immune._detectors = [detector]
    immune._trained = True

    result = immune.check(_crash_screen())
    assert result.is_anomaly
    assert result.matching_detectors == 1

    result = immune.check(_normal_screen())
    assert not result.is_anomaly
