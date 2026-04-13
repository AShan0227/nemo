"""ADB device controller — wraps adbutils for phone interaction."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

    from src.config.settings import DeviceConfig

# Validation patterns for shell safety
_KEYCODE_RE = re.compile(r"^[A-Z_0-9]+$")
_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9._]+$")


def _escape_shell_text(text: str) -> str:
    """Escape text for safe use in Android shell commands.

    Wraps in single quotes and escapes embedded single quotes.
    """
    return "'" + text.replace("'", "'\\''") + "'"


class ADBController:
    """Low-level Android device control via ADB."""

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device = None
        self._screen_size: tuple[int, int] | None = None

    async def connect(self) -> None:
        """Connect to device (auto-detect or by serial)."""
        import adbutils

        client = adbutils.AdbClient(self._config.adb_host, self._config.adb_port)
        devices = client.device_list()

        if not devices:
            raise ConnectionError("No Android device found. Check ADB connection.")

        if self._config.serial:
            self._device = client.device(self._config.serial)
        else:
            self._device = devices[0]

        info = self._device.info
        logger.info(f"Connected to {info.get('model', 'unknown')} ({self._device.serial})")

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._device

    async def screenshot(self) -> Image.Image:
        """Capture current screen as PIL Image."""
        from PIL import Image
        import io

        png_bytes = await asyncio.to_thread(self.device.screenshot)
        return png_bytes if isinstance(png_bytes, Image.Image) else Image.open(io.BytesIO(png_bytes))

    async def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates."""
        logger.debug(f"tap({x}, {y})")
        await asyncio.to_thread(self.device.click, x, y)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from (x1,y1) to (x2,y2)."""
        logger.debug(f"swipe({x1},{y1} -> {x2},{y2}, {duration_ms}ms)")
        await asyncio.to_thread(self.device.swipe, x1, y1, x2, y2, duration_ms / 1000)

    async def get_screen_size(self) -> tuple[int, int]:
        """Get device screen resolution (width, height). Cached after first call."""
        if self._screen_size is None:
            output = await asyncio.to_thread(self.device.shell, "wm size")
            match = re.search(r"(\d+)x(\d+)", output)
            if match:
                self._screen_size = (int(match.group(1)), int(match.group(2)))
            else:
                logger.warning("Could not parse screen size, using 1080x1920 default")
                self._screen_size = (1080, 1920)
        return self._screen_size

    async def input_text(self, text: str) -> None:
        """Input text via ADB shell (safely escaped)."""
        logger.debug(f"input_text({text!r})")
        escaped = _escape_shell_text(text)
        await asyncio.to_thread(self.device.shell, f"input text {escaped}")

    async def press_key(self, keycode: str) -> None:
        """Press a key (BACK, HOME, ENTER, etc.)."""
        if not _KEYCODE_RE.match(keycode):
            raise ValueError(f"Invalid keycode format: {keycode!r}")
        logger.debug(f"press_key({keycode})")
        await asyncio.to_thread(self.device.shell, f"input keyevent {keycode}")

    async def press_back(self) -> None:
        await self.press_key("KEYCODE_BACK")

    async def press_home(self) -> None:
        await self.press_key("KEYCODE_HOME")

    async def get_ui_hierarchy(self) -> str:
        """Dump UI hierarchy XML via uiautomator."""
        xml = await asyncio.to_thread(self.device.shell, "uiautomator dump /dev/tty")
        return xml

    async def get_current_activity(self) -> str:
        """Get current foreground activity."""
        result = await asyncio.to_thread(
            self.device.shell,
            "dumpsys activity activities | grep mResumedActivity",
        )
        return result.strip()

    async def launch_app(self, package: str) -> None:
        """Launch an app by package name."""
        if not _PACKAGE_RE.match(package):
            raise ValueError(f"Invalid package name: {package!r}")
        logger.info(f"Launching {package}")
        await asyncio.to_thread(
            self.device.shell,
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1",
        )
