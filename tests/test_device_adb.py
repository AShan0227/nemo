"""Tests for ADB device controller enhancements."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src.device.adb import ADBController


class FakeDevice:
    serial = "FAKE123"
    info = {"model": "FakePhone"}

    def __init__(self) -> None:
        self.shell_outputs: dict[str, str] = {}
        self.shell_history: list[str] = []
        self.click_history: list[tuple[int, int]] = []
        self.swipe_history: list[tuple[int, int, int, int, float]] = []
        self.screenshot_color: tuple[int, int, int] = (0, 0, 0)

    def screenshot(self) -> Image.Image:
        return Image.new("RGB", (100, 200), self.screenshot_color)

    def click(self, x: int, y: int) -> None:
        self.click_history.append((x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float) -> None:
        self.swipe_history.append((x1, y1, x2, y2, duration))

    def shell(self, command: str) -> str:
        self.shell_history.append(command)
        return self.shell_outputs.get(command, "")


def make_controller(tmp_path: Path, device: FakeDevice) -> ADBController:
    config = SimpleNamespace(
        serial=None,
        adb_host="127.0.0.1",
        adb_port=5037,
        screenshot_dir=str(tmp_path),
    )
    controller = ADBController(config)  # type: ignore[arg-type]
    controller._device = device  # type: ignore[attr-defined]
    return controller


@pytest.mark.asyncio
async def test_get_screen_size_uses_cache(tmp_path: Path):
    device = FakeDevice()
    device.shell_outputs["wm size"] = "Physical size: 1080x2400"
    controller = make_controller(tmp_path, device)

    assert await controller.get_screen_size() == (1080, 2400)
    assert await controller.get_screen_size() == (1080, 2400)
    assert device.shell_history.count("wm size") == 1


@pytest.mark.asyncio
async def test_normalize_point_supports_ratio_and_absolute(tmp_path: Path):
    device = FakeDevice()
    device.shell_outputs["wm size"] = "Physical size: 1080x2400"
    controller = make_controller(tmp_path, device)

    assert await controller.normalize_point(0.5, 0.25) == (540, 600)
    assert await controller.normalize_point(5000, -10) == (1079, 0)


@pytest.mark.asyncio
async def test_gesture_commands(tmp_path: Path):
    device = FakeDevice()
    controller = make_controller(tmp_path, device)

    await controller.long_press(10, 20, duration_ms=900)
    await controller.double_tap(5, 6, interval_ms=0)
    await controller.pinch_zoom(300, 500, distance=200, zoom_in=True, duration_ms=320)

    assert "input swipe 10 20 10 20 900" in device.shell_history
    assert device.click_history[-2:] == [(5, 6), (5, 6)]
    assert any(
        command.startswith("sh -c ") and "& wait" in command
        for command in device.shell_history
    )


@pytest.mark.asyncio
async def test_installed_app_and_foreground_package(tmp_path: Path):
    device = FakeDevice()
    device.shell_outputs[
        "pm list packages com.demo.app"
    ] = "package:com.demo.app\npackage:other.app"
    device.shell_outputs[
        "dumpsys activity activities | grep mResumedActivity"
    ] = "mResumedActivity: ActivityRecord{1 u0 com.android.settings/.Settings t91}"
    controller = make_controller(tmp_path, device)

    assert await controller.is_app_installed("com.demo.app")
    assert not await controller.is_app_installed("com.missing.app")
    assert await controller.get_foreground_app() == "com.android.settings"


@pytest.mark.asyncio
async def test_save_and_compare_screenshots(tmp_path: Path):
    device = FakeDevice()
    controller = make_controller(tmp_path, device)

    before = await controller.save_screenshot("before.png")
    device.screenshot_color = (255, 255, 255)
    after = await controller.save_screenshot("after.png")

    diff_ratio = await controller.screenshot_diff_ratio(before, after)
    same_ratio = await controller.screenshot_diff_ratio(before, before)
    changed = await controller.screenshot_changed(before, after, threshold=0.20)

    assert before.exists()
    assert after.exists()
    assert diff_ratio > 0.95
    assert same_ratio == pytest.approx(0.0, abs=1e-6)
    assert changed


class OfflineOnceDevice(FakeDevice):
    def __init__(self) -> None:
        super().__init__()
        self._offline_once = True

    def click(self, x: int, y: int) -> None:
        if self._offline_once:
            self._offline_once = False
            raise RuntimeError("device offline")
        super().click(x, y)


@pytest.mark.asyncio
async def test_reconnect_retry_on_transient_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    device = OfflineOnceDevice()
    controller = make_controller(tmp_path, device)
    reconnect_count = {"count": 0}

    async def fake_reconnect() -> None:
        reconnect_count["count"] += 1

    monkeypatch.setattr(controller, "reconnect", fake_reconnect)
    await controller.tap(7, 8)

    assert reconnect_count["count"] == 1
    assert device.click_history == [(7, 8)]
