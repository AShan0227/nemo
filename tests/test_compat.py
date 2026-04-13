"""Tests for device compatibility checker (unit tests without real device)."""

from src.device.compat import CompatCheck, CompatReport


def test_compat_check_pass():
    c = CompatCheck(name="test", passed=True, detail="ok", elapsed_ms=10.0)
    assert c.passed


def test_compat_report_all_passed():
    r = CompatReport(
        device_model="Pixel",
        checks=[
            CompatCheck("a", True, "ok"),
            CompatCheck("b", True, "ok"),
        ],
    )
    assert r.all_passed
    assert r.pass_count == 2


def test_compat_report_partial_fail():
    r = CompatReport(
        checks=[
            CompatCheck("a", True, "ok"),
            CompatCheck("b", False, "err"),
        ],
    )
    assert not r.all_passed
    assert r.pass_count == 1


def test_compat_report_summary():
    r = CompatReport(
        device_model="Pixel 6",
        android_version="13",
        screen_size=(1080, 2400),
        checks=[
            CompatCheck("screen", True, "1080x2400"),
            CompatCheck("ui_dump", False, "timeout"),
        ],
    )
    s = r.summary()
    assert "Pixel 6" in s
    assert "PASS" in s
    assert "FAIL" in s


def test_compat_report_to_dict():
    r = CompatReport(
        device_model="Test",
        android_version="14",
        screen_size=(1080, 1920),
        checks=[CompatCheck("x", True, "ok", 5.0)],
    )
    d = r.to_dict()
    assert d["all_passed"] is True
    assert len(d["checks"]) == 1
    assert d["checks"][0]["elapsed_ms"] == 5.0


def test_compat_report_empty():
    r = CompatReport()
    assert r.all_passed  # vacuously true
    assert r.pass_count == 0
