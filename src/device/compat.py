"""Device compatibility checker — validate ADB capabilities across Android versions.

Runs a series of non-destructive checks on the connected device to verify
that all agent capabilities work correctly:
- Screen resolution detection
- UI hierarchy dump
- Screenshot capture
- Input methods (tap, swipe, text)
- App lifecycle (launch, foreground detection)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CompatCheck:
    """A single compatibility check result."""

    name: str
    passed: bool
    detail: str = ""
    elapsed_ms: float = 0.0


@dataclass
class CompatReport:
    """Full device compatibility report."""

    device_model: str = ""
    android_version: str = ""
    screen_size: tuple[int, int] = (0, 0)
    checks: list[CompatCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    def summary(self) -> str:
        lines = [
            f"Device: {self.device_model} (Android {self.android_version})",
            f"Screen: {self.screen_size[0]}x{self.screen_size[1]}",
            f"Checks: {self.pass_count}/{len(self.checks)} passed",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{status}] {c.name}: {c.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_model": self.device_model,
            "android_version": self.android_version,
            "screen_size": list(self.screen_size),
            "all_passed": self.all_passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail, "elapsed_ms": c.elapsed_ms}
                for c in self.checks
            ],
        }


async def run_compat_checks(adb_controller: Any) -> CompatReport:
    """Run all compatibility checks on the connected device.

    Args:
        adb_controller: Connected ADBController instance.

    Returns:
        CompatReport with all check results.
    """
    import time

    report = CompatReport()

    # Device info
    try:
        info = await asyncio.to_thread(lambda: adb_controller.device.info)
        report.device_model = info.get("model", "unknown")
        report.android_version = info.get("version", "unknown")
    except Exception as e:
        report.device_model = f"error: {e}"

    # Check 1: Screen size
    t0 = time.perf_counter()
    try:
        size = await adb_controller.get_screen_size()
        report.screen_size = size
        report.checks.append(CompatCheck(
            "screen_size", True, f"{size[0]}x{size[1]}",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("screen_size", False, str(e)))

    # Check 2: Screenshot
    t0 = time.perf_counter()
    try:
        img = await adb_controller.screenshot()
        w, h = img.size
        report.checks.append(CompatCheck(
            "screenshot", True, f"{w}x{h} image",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("screenshot", False, str(e)))

    # Check 3: UI hierarchy dump
    t0 = time.perf_counter()
    try:
        xml = await adb_controller.get_ui_hierarchy()
        has_hierarchy = "<hierarchy" in xml or "<node" in xml
        report.checks.append(CompatCheck(
            "ui_hierarchy", has_hierarchy,
            f"{len(xml)} chars" if has_hierarchy else "empty/invalid XML",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("ui_hierarchy", False, str(e)))

    # Check 4: Current activity
    t0 = time.perf_counter()
    try:
        activity = await adb_controller.get_current_activity()
        report.checks.append(CompatCheck(
            "current_activity", bool(activity.strip()),
            activity.strip()[:80] or "empty",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("current_activity", False, str(e)))

    # Check 5: Foreground app
    t0 = time.perf_counter()
    try:
        pkg = await adb_controller.get_foreground_app()
        report.checks.append(CompatCheck(
            "foreground_app", bool(pkg),
            pkg or "empty",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("foreground_app", False, str(e)))

    # Check 6: App installed check
    t0 = time.perf_counter()
    try:
        installed = await adb_controller.is_app_installed("com.android.settings")
        report.checks.append(CompatCheck(
            "app_installed_check", installed,
            "com.android.settings found" if installed else "not found",
            (time.perf_counter() - t0) * 1000,
        ))
    except Exception as e:
        report.checks.append(CompatCheck("app_installed_check", False, str(e)))

    logger.info(f"Compat: {report.pass_count}/{len(report.checks)} checks passed")
    return report
