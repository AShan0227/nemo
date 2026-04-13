"""Tests for homeostatic regulation."""

from src.research.homeostasis import HomeostasisRegulator, RegulatedVariable


def test_drive_computation():
    var = RegulatedVariable(name="test", setpoint=0.9)
    var.update(0.5)
    assert abs(var.drive - 0.4) < 0.001
    assert var.normalized_drive > 0  # below setpoint = positive drive


def test_no_adjustment_when_healthy():
    reg = HomeostasisRegulator()
    reg.update_metrics(success_rate=0.95, response_latency_ms=1500, error_rate=0.02)
    adjustments = reg.get_adjustments()
    assert len(adjustments) == 0


def test_verify_adjustment_on_low_success():
    reg = HomeostasisRegulator()
    reg.update_metrics(success_rate=0.5)
    adjustments = reg.get_adjustments()
    names = [a.name for a in adjustments]
    assert "verify_actions" in names


def test_reduce_prompt_on_high_latency():
    reg = HomeostasisRegulator()
    reg.update_metrics(response_latency_ms=5000)
    adjustments = reg.get_adjustments()
    names = [a.name for a in adjustments]
    assert "reduce_prompt" in names


def test_increase_delay_on_high_errors():
    reg = HomeostasisRegulator()
    reg.update_metrics(error_rate=0.3)
    adjustments = reg.get_adjustments()
    names = [a.name for a in adjustments]
    assert "increase_delay" in names


def test_health_summary():
    reg = HomeostasisRegulator()
    reg.update_metrics(success_rate=0.8)
    summary = reg.health_summary
    assert "success_rate" in summary
    assert summary["success_rate"]["current"] == 0.8
    assert summary["success_rate"]["setpoint"] == 0.9


def test_trend_tracking():
    var = RegulatedVariable(name="test", setpoint=0.9)
    var.update(0.5)
    var.update(0.6)
    var.update(0.7)
    assert var.trend > 0  # improving
