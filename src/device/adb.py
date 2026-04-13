"""ADB device controller — wraps adbutils for phone interaction."""

from __future__ import annotations

import asyncio
import io
import re
import shlex
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

    from src.config.settings import DeviceConfig

# Validation patterns for shell safety
_KEYCODE_RE = re.compile(r"^[A-Z_0-9]+$")
_PACKAGE_RE = re.compile(r"^[a-zA-Z0-9._]+$")


def _escape_shell_text(text: str) -> str:
    """Escape text for safe use in Android shell commands."""
    return "'" + text.replace("'", "'\\''") + "'"


class ADBController:
    """Low-level Android device control via ADB."""

    _CONNECTION_ERROR_MARKERS = (
        "device offline",
        "no devices/emulators found",
        "device not found",
        "broken pipe",
        "connection reset",
        "closed",
        "not connected",
    )

    def __init__(self, config: DeviceConfig) -> None:
        self._config = config
        self._client: Any | None = None
        self._device: Any | None = None
        self._screen_size_cache: tuple[int, int] | None = None

    async def connect(self) -> None:
        """Connect to device (auto-detect or by serial)."""
        import adbutils

        client = adbutils.AdbClient(self._config.adb_host, self._config.adb_port)
        devices = await asyncio.to_thread(client.device_list)

        if not devices:
            raise ConnectionError("No Android device found. Check ADB connection.")

        if self._config.serial:
            device = await asyncio.to_thread(client.device, self._config.serial)
        else:
            device = devices[0]

        self._client = client
        self._device = device
        self._screen_size_cache = None

        info = await asyncio.to_thread(lambda: self.device.info)
        logger.info(f"Connected to {info.get('model', 'unknown')} ({self.device.serial})")

    async def reconnect(self) -> None:
        """Reconnect to device after transient transport failures."""
        logger.warning("Reconnecting ADB device...")
        self._device = None
        self._screen_size_cache = None
        await self.connect()

    @property
    def device(self) -> Any:
        if self._device is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._device

    async def screenshot(self) -> Image.Image:
        """Capture current screen as PIL Image."""
        from PIL import Image

        png_data = await self._run_device_call(self.device.screenshot, "screenshot")
        if isinstance(png_data, Image.Image):
            return png_data.copy()
        with Image.open(io.BytesIO(png_data)) as image:
            return image.convert("RGB")

    async def save_screenshot(self, filename: str | None = None) -> Path:
        """Capture and save a screenshot under configured screenshot directory."""
        image = await self.screenshot()
        screenshot_dir = Path(self._config.screenshot_dir)
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        path = screenshot_dir / filename
        await asyncio.to_thread(image.save, path)
        return path

    async def screenshot_diff_ratio(self, before: str | Path, after: str | Path) -> float:
        """Return normalized visual diff ratio (0.0 same, 1.0 fully different)."""
        return await asyncio.to_thread(self._compute_image_diff_ratio, Path(before), Path(after))

    async def screenshot_changed(
        self, before: str | Path, after: str | Path, threshold: float = 0.01,
    ) -> bool:
        """Check whether two screenshots differ beyond threshold."""
        ratio = await self.screenshot_diff_ratio(before, after)
        return ratio >= threshold

    async def tap(self, x: int, y: int) -> None:
        """Tap at screen coordinates."""
        logger.debug(f"tap({x}, {y})")
        await self._run_device_call(lambda: self.device.click(x, y), "tap")

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        """Swipe from (x1,y1) to (x2,y2)."""
        logger.debug(f"swipe({x1},{y1} -> {x2},{y2}, {duration_ms}ms)")
        await self._run_device_call(
            lambda: self.device.swipe(x1, y1, x2, y2, duration_ms / 1000), "swipe",
        )

    async def long_press(self, x: int, y: int, duration_ms: int = 800) -> None:
        """Long press by swiping from a point to itself."""
        logger.debug(f"long_press({x}, {y}, {duration_ms}ms)")
        await self._shell_command(f"input swipe {x} {y} {x} {y} {duration_ms}")

    async def double_tap(self, x: int, y: int, interval_ms: int = 120) -> None:
        """Tap twice with a short configurable interval."""
        logger.debug(f"double_tap({x}, {y}, interval={interval_ms}ms)")
        await self.tap(x, y)
        await asyncio.sleep(interval_ms / 1000)
        await self.tap(x, y)

    async def pinch_zoom(
        self, center_x: int, center_y: int, *,
        distance: int = 220, zoom_in: bool = True, duration_ms: int = 350,
    ) -> None:
        """Best-effort pinch/zoom gesture using concurrent shell swipes."""
        if distance <= 0:
            raise ValueError("distance must be > 0")
        offset = max(20, distance // 2)
        near = 24
        if zoom_in:
            left_start, left_end = center_x - near, center_x - offset
            right_start, right_end = center_x + near, center_x + offset
        else:
            left_start, left_end = center_x - offset, center_x - near
            right_start, right_end = center_x + offset, center_x + near
        gesture_cmd = (
            f"input swipe {left_start} {center_y} {left_end} {center_y} {duration_ms} & "
            f"input swipe {right_start} {center_y} {right_end} {center_y} {duration_ms} & wait"
        )
        await self._shell_command(f"sh -c {shlex.quote(gesture_cmd)}")

    async def get_screen_size(self) -> tuple[int, int]:
        """Read and cache device screen resolution as (width, height)."""
        if self._screen_size_cache is not None:
            return self._screen_size_cache
        raw = await self._shell_command("wm size")
        parsed = self._parse_screen_size_output(raw)
        if parsed is None:
            screenshot = await self.screenshot()
            parsed = screenshot.size
        self._screen_size_cache = parsed
        return parsed

    async def input_text(self, text: str) -> None:
        """Input text via ADB shell (safely escaped)."""
        logger.debug(f"input_text({text!r})")
        escaped = text.replace(" ", "%s")
        await self._shell_command(f"input text {shlex.quote(escaped)}")

    async def press_key(self, keycode: str) -> None:
        """Press a key (BACK, HOME, ENTER, etc.)."""
        if not _KEYCODE_RE.match(keycode):
            raise ValueError(f"Invalid keycode format: {keycode!r}")
        logger.debug(f"press_key({keycode})")
        await self._shell_command(f"input keyevent {keycode}")

    async def press_back(self) -> None:
        await self.press_key("KEYCODE_BACK")

    async def press_home(self) -> None:
        await self.press_key("KEYCODE_HOME")

    async def get_ui_hierarchy(self) -> str:
        """Dump UI hierarchy XML via uiautomator."""
        return await self._shell_command("uiautomator dump /dev/tty")

    async def get_current_activity(self) -> str:
        """Get current foreground activity."""
        activity_line = await self._shell_command(
            "dumpsys activity activities | grep mResumedActivity",
        )
        if activity_line.strip():
            return activity_line.strip()
        fallback = await self._shell_command("dumpsys window windows | grep mCurrentFocus")
        return fallback.strip()

    async def get_foreground_app(self) -> str:
        """Return current foreground app package name if available."""
        activity = await self.get_current_activity()
        match = re.search(r"([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.$]+", activity)
        return match.group(1) if match else ""

    async def launch_app(self, package: str) -> None:
        """Launch an app by package name."""
        if not _PACKAGE_RE.match(package):
            raise ValueError(f"Invalid package name: {package!r}")
        logger.info(f"Launching {package}")
        await self._shell_command(
            f"monkey -p {package} -c android.intent.category.LAUNCHER 1",
        )

    async def is_app_installed(self, package: str) -> bool:
        """Check whether an app package is installed on device."""
        output = await self._shell_command(f"pm list packages {package}")
        package_line = f"package:{package}"
        return any(line.strip() == package_line for line in output.splitlines())

    async def normalize_point(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized coordinates [0,1] to absolute, or clamp absolute input."""
        width, height = await self.get_screen_size()
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            abs_x = int(round(x * (width - 1)))
            abs_y = int(round(y * (height - 1)))
        else:
            abs_x = int(round(x))
            abs_y = int(round(y))
        return (min(max(abs_x, 0), width - 1), min(max(abs_y, 0), height - 1))

    async def swipe_normalized(
        self, x1: float, y1: float, x2: float, y2: float, duration_ms: int = 300,
    ) -> None:
        """Swipe using normalized coordinates."""
        ax1, ay1 = await self.normalize_point(x1, y1)
        ax2, ay2 = await self.normalize_point(x2, y2)
        await self.swipe(ax1, ay1, ax2, ay2, duration_ms=duration_ms)

    async def _shell_command(self, command: str) -> str:
        result = await self._run_device_call(lambda: self.device.shell(command), f"shell:{command}")
        return str(result)

    async def _run_device_call(self, fn: Callable[[], Any], op_name: str) -> Any:
        for attempt in range(2):
            try:
                return await asyncio.to_thread(fn)
            except Exception as exc:
                can_retry = attempt == 0 and self._is_connection_error(exc)
                if not can_retry:
                    raise
                logger.warning(f"ADB op `{op_name}` failed ({exc}), retrying after reconnect")
                await self.reconnect()
        raise RuntimeError("Unreachable retry branch in _run_device_call")

    @classmethod
    def _is_connection_error(cls, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(marker in message for marker in cls._CONNECTION_ERROR_MARKERS)

    @staticmethod
    def _parse_screen_size_output(output: str) -> tuple[int, int] | None:
        override = re.search(r"Override size:\s*(\d+)x(\d+)", output)
        if override:
            return (int(override.group(1)), int(override.group(2)))
        physical = re.search(r"Physical size:\s*(\d+)x(\d+)", output)
        if physical:
            return (int(physical.group(1)), int(physical.group(2)))
        generic = re.search(r"(\d+)x(\d+)", output)
        if generic:
            return (int(generic.group(1)), int(generic.group(2)))
        return None

    @staticmethod
    def _compute_image_diff_ratio(before: Path, after: Path) -> float:
        from PIL import Image, ImageChops, ImageStat

        with Image.open(before) as left_image, Image.open(after) as right_image:
            left = left_image.convert("RGB")
            right = right_image.convert("RGB")
            if left.size != right.size:
                right = right.resize(left.size)
            diff = ImageChops.difference(left, right)
            stats = ImageStat.Stat(diff)
            channel_mean = sum(stats.mean) / len(stats.mean)
            return float(channel_mean / 255.0)

    @staticmethod
    def compute_phash(image: "Image.Image", hash_size: int = 16) -> int:
        """Compute perceptual hash (pHash) for fast image similarity.

        Uses gradient-based approach: resize -> grayscale -> row/col gradients -> threshold.
        Returns integer hash for Hamming distance comparison.
        """
        import numpy as np

        # Resize to hash_size+1 to compute gradients
        small = image.convert("L").resize((hash_size + 1, hash_size + 1))
        pixels = np.array(small, dtype=np.float64)
        # Horizontal gradient: pixel[x] > pixel[x+1]
        h_bits = (pixels[:, 1:] > pixels[:, :-1])[:hash_size, :hash_size].flatten()
        # Vertical gradient
        v_bits = (pixels[1:, :] > pixels[:-1, :])[:hash_size, :hash_size].flatten()
        all_bits = np.concatenate([h_bits, v_bits])
        hash_val = 0
        for bit in all_bits:
            hash_val = (hash_val << 1) | int(bit)
        return hash_val

    @staticmethod
    def phash_distance(hash_a: int, hash_b: int) -> int:
        """Hamming distance between two perceptual hashes."""
        return bin(hash_a ^ hash_b).count("1")

    @staticmethod
    def phash_similarity(hash_a: int, hash_b: int, hash_size: int = 16) -> float:
        """Similarity score (0.0 to 1.0) between two pHashes."""
        total_bits = 2 * hash_size * hash_size  # h_grad + v_grad
        dist = bin(hash_a ^ hash_b).count("1")
        return 1.0 - (dist / total_bits)

    async def screenshot_diff_ratio_phash(
        self, before: str | Path, after: str | Path, hash_size: int = 16,
    ) -> float:
        """Compare screenshots using perceptual hash (faster than pixel diff)."""
        from PIL import Image

        def _compute():
            with Image.open(str(before)) as img_a, Image.open(str(after)) as img_b:
                ha = ADBController.compute_phash(img_a, hash_size)
                hb = ADBController.compute_phash(img_b, hash_size)
                return 1.0 - ADBController.phash_similarity(ha, hb, hash_size)

        return await asyncio.to_thread(_compute)

    async def screenshot_diff_region(
        self,
        before: str | Path,
        after: str | Path,
        region: tuple[int, int, int, int],
    ) -> float:
        """Compare only a specific region of two screenshots.

        Args:
            before: Path to first screenshot.
            after: Path to second screenshot.
            region: (left, top, right, bottom) crop region to compare.

        Returns:
            Diff ratio (0.0 same, 1.0 fully different) for the specified region.
        """
        from PIL import Image, ImageChops, ImageStat

        def _compute():
            with Image.open(str(before)) as img_a, Image.open(str(after)) as img_b:
                crop_a = img_a.convert("RGB").crop(region)
                crop_b = img_b.convert("RGB").crop(region)
                if crop_a.size != crop_b.size:
                    crop_b = crop_b.resize(crop_a.size)
                diff = ImageChops.difference(crop_a, crop_b)
                stats = ImageStat.Stat(diff)
                channel_mean = sum(stats.mean) / len(stats.mean)
                return float(channel_mean / 255.0)

        return await asyncio.to_thread(_compute)
