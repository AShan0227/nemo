"""ADB device controller — wraps adbutils for phone interaction."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

    from src.config.settings import DeviceConfig


class ADBController:
    """Low-level Android device control via ADB.

    Features beyond basic ADB:
    - Auto-reconnect on connection drop
    - Screen resolution detection & coordinate normalization
    - Extended gestures (long press, double tap, pinch zoom)
    - App install/foreground state detection
    - Screenshot save & comparison
    """

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._device = None
        self._client = None
        self._screen_width: int = 0
        self._screen_height: int = 0
        self._reconnect_attempts = 3
        self._reconnect_delay = 2.0

    # ---- Connection Management ----

    async def connect(self) -> None:
        """Connect to device (auto-detect or by serial)."""
        import adbutils

        self._client = adbutils.AdbClient(self._config.adb_host, self._config.adb_port)
        await self._connect_device()
        await self._detect_resolution()

    async def _connect_device(self) -> None:
        """Internal connect with device selection."""
        devices = self._client.device_list()

        if not devices:
            raise ConnectionError("No Android device found. Check ADB connection.")

        if self._config.serial:
            self._device = self._client.device(self._config.serial)
        else:
            self._device = devices[0]

        info = self._device.info
        logger.info(f"Connected to {info.get('model', 'unknown')} ({self._device.serial})")

    async def reconnect(self) -> bool:
        """Attempt to reconnect to the device after a connection drop."""
        for attempt in range(1, self._reconnect_attempts + 1):
            logger.warning(f"Reconnect attempt {attempt}/{self._reconnect_attempts}...")
            try:
                await asyncio.sleep(self._reconnect_delay)
                await self._connect_device()
                await self._detect_resolution()
                logger.info("Reconnected successfully")
                return True
            except Exception as e:
                logger.error(f"Reconnect attempt {attempt} failed: {e}")
        logger.error("All reconnect attempts exhausted")
        return False

    async def _ensure_connected(self) -> None:
        """Verify connection is alive; reconnect if not."""
        if self._device is None:
            raise RuntimeError("Not connected. Call connect() first.")
        try:
            await asyncio.to_thread(self._device.shell, "echo ping")
        except Exception:
            logger.warning("Device connection lost, attempting reconnect...")
            if not await self.reconnect():
                raise ConnectionError("Cannot reach device after reconnect attempts.")

    @property
    def device(self):
        if self._device is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._device

    # ---- Screen Resolution ----

    async def _detect_resolution(self) -> None:
        """Detect device screen resolution."""
        result = await asyncio.to_thread(self.device.shell, "wm size")
        # Output: "Physical size: 1080x1920" or "Override size: ..."
        for line in result.strip().splitlines():
            if "size:" in line.lower():
                parts = line.split(":")[-1].strip().split("x")
                if len(parts) == 2:
                    self._screen_width = int(parts[0])
                    self._screen_height = int(parts[1])
                    logger.info(f"Screen resolution: {self._screen_width}x{self._screen_height}")
                    return
        # Fallback
        self._screen_width = 1080
        self._screen_height = 1920
        logger.warning("Could not detect resolution, using 1080x1920 default")

    @property
    def screen_size(self) -> tuple[int, int]:
        """Returns (width, height) of the device screen."""
        return (self._screen_width, self._screen_height)

    def normalize_coords(self, x: int, y: int, ref_w: int = 1080, ref_h: int = 1920) -> tuple[int, int]:
        """Scale coordinates from a reference resolution to the actual device resolution."""
        if ref_w == self._screen_width and ref_h == self._screen_height:
            return (x, y)
        nx = int(x * self._screen_width / ref_w)
        ny = int(y * self._screen_height / ref_h)
        return (nx, ny)

    # ---- Basic Input ----

    async def screenshot(self) -> Image.Image:
        """Capture current screen as PIL Image."""
        from PIL import Image
        import io

        await self._ensure_connected()
        png_bytes = await asyncio.to_thread(self.device.screenshot)
        return png_bytes if isinstance(png_bytes, Image.Image) else Image.open(io.BytesIO(png_bytes))

    async def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates."""
        await self._ensure_connected()
        logger.debug(f"tap({x}, {y})")
        await asyncio.to_thread(self.device.click, x, y)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from (x1,y1) to (x2,y2)."""
        await self._ensure_connected()
        logger.debug(f"swipe({x1},{y1} -> {x2},{y2}, {duration_ms}ms)")
        await asyncio.to_thread(self.device.swipe, x1, y1, x2, y2, duration_ms / 1000)

    async def input_text(self, text: str) -> None:
        """Input text via ADB shell."""
        await self._ensure_connected()
        logger.debug(f"input_text({text!r})")
        # Escape special characters for shell
        escaped = text.replace("'", "'\\''")
        await asyncio.to_thread(self.device.shell, f"input text '{escaped}'")

    async def press_key(self, keycode: str) -> None:
        """Press a key (BACK, HOME, ENTER, etc.)."""
        await self._ensure_connected()
        logger.debug(f"press_key({keycode})")
        await asyncio.to_thread(self.device.shell, f"input keyevent {keycode}")

    async def press_back(self) -> None:
        await self.press_key("KEYCODE_BACK")

    async def press_home(self) -> None:
        await self.press_key("KEYCODE_HOME")

    # ---- Extended Gestures ----

    async def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        """Long press at coordinates (swipe to same point with duration)."""
        await self._ensure_connected()
        logger.debug(f"long_press({x}, {y}, {duration_ms}ms)")
        await asyncio.to_thread(
            self.device.shell,
            f"input swipe {x} {y} {x} {y} {duration_ms}",
        )

    async def double_tap(self, x: int, y: int, interval_ms: int = 100) -> None:
        """Double tap at coordinates."""
        await self._ensure_connected()
        logger.debug(f"double_tap({x}, {y})")
        await asyncio.to_thread(self.device.click, x, y)
        await asyncio.sleep(interval_ms / 1000)
        await asyncio.to_thread(self.device.click, x, y)

    async def pinch_zoom(self, cx: int, cy: int, zoom_in: bool = True, scale: float = 0.5) -> None:
        """Pinch zoom in/out centered at (cx, cy).

        Uses two-finger gesture approximation via input commands.
        """
        await self._ensure_connected()
        offset = int(min(self._screen_width, self._screen_height) * scale / 2)
        direction = "in" if zoom_in else "out"
        logger.debug(f"pinch_zoom({direction}, center=({cx},{cy}), scale={scale})")

        if zoom_in:
            # Two fingers move apart
            x1_start, y1_start = cx - offset // 4, cy - offset // 4
            x1_end, y1_end = cx - offset, cy - offset
            x2_start, y2_start = cx + offset // 4, cy + offset // 4
            x2_end, y2_end = cx + offset, cy + offset
        else:
            # Two fingers move together
            x1_start, y1_start = cx - offset, cy - offset
            x1_end, y1_end = cx - offset // 4, cy - offset // 4
            x2_start, y2_start = cx + offset, cy + offset
            x2_end, y2_end = cx + offset // 4, cy + offset // 4

        # Execute as two sequential swipes (approximation)
        await asyncio.to_thread(
            self.device.shell,
            f"input swipe {x1_start} {y1_start} {x1_end} {y1_end} 500",
        )
        await asyncio.to_thread(
            self.device.shell,
            f"input swipe {x2_start} {y2_start} {x2_end} {y2_end} 500",
        )

    # ---- Screen & UI ----

    async def get_ui_hierarchy(self) -> str:
        """Dump UI hierarchy XML via uiautomator."""
        await self._ensure_connected()
        xml = await asyncio.to_thread(self.device.shell, "uiautomator dump /dev/tty")
        return xml

    async def get_current_activity(self) -> str:
        """Get current foreground activity."""
        await self._ensure_connected()
        result = await asyncio.to_thread(
            self.device.shell,
            "dumpsys activity activities | grep mResumedActivity",
        )
        return result.strip()

    # ---- App Management ----

    async def launch_app(self, package: str) -> None:
        """Launch an app by package name."""
        await self._ensure_connected()
        logger.info(f"Launching {package}")
        await asyncio.to_thread(
            self.device.shell,
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1",
        )

    async def is_app_installed(self, package: str) -> bool:
        """Check if an app is installed on the device."""
        await self._ensure_connected()
        result = await asyncio.to_thread(self.device.shell, f"pm list packages {package}")
        return f"package:{package}" in result

    async def get_foreground_package(self) -> str:
        """Get the package name of the current foreground app."""
        await self._ensure_connected()
        result = await asyncio.to_thread(
            self.device.shell,
            "dumpsys activity recents | grep 'Recent #0' | grep -oP 'A=\\K[^ ]*'",
        )
        pkg = result.strip()
        if not pkg:
            # Fallback for different Android versions
            result = await asyncio.to_thread(
                self.device.shell,
                "dumpsys window | grep mCurrentFocus",
            )
            parts = result.strip().split("/")
            if parts:
                pkg = parts[0].split()[-1] if " " in parts[0] else parts[0]
        return pkg

    # ---- Screenshot Save & Compare ----

    async def save_screenshot(self, filename: str | None = None) -> Path:
        """Capture and save screenshot to disk. Returns the file path."""
        img = await self.screenshot()
        save_dir = Path(self._config.screenshot_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            import time
            filename = f"screen_{int(time.time() * 1000)}.png"

        path = save_dir / filename
        img.save(str(path))
        logger.debug(f"Screenshot saved: {path}")
        return path

    @staticmethod
    def compare_screenshots(img_a: Image.Image, img_b: Image.Image) -> float:
        """Compare two screenshots and return similarity score (0.0 to 1.0).

        Uses perceptual hash comparison for speed.
        """
        import numpy as np

        # Resize both to small thumbnail for fast comparison
        size = (64, 64)
        a = img_a.convert("L").resize(size)
        b = img_b.convert("L").resize(size)
        arr_a = np.array(a, dtype=np.float32)
        arr_b = np.array(b, dtype=np.float32)

        # Normalized cross-correlation
        a_norm = arr_a - arr_a.mean()
        b_norm = arr_b - arr_b.mean()
        denom = np.linalg.norm(a_norm) * np.linalg.norm(b_norm)
        if denom == 0:
            return 1.0 if np.array_equal(arr_a, arr_b) else 0.0
        similarity = float(np.sum(a_norm * b_norm) / denom)
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def screenshot_hash(img: Image.Image) -> str:
        """Compute a perceptual hash for quick screen dedup."""
        import numpy as np

        small = img.convert("L").resize((16, 16))
        arr = np.array(small)
        avg = arr.mean()
        bits = (arr > avg).flatten()
        hex_str = "".join(str(int(b)) for b in bits)
        return hashlib.md5(hex_str.encode()).hexdigest()[:12]
