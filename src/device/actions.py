"""High-level action primitives built on ADB controller."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class ActionType(Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    DOUBLE_TAP = "double_tap"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    LAUNCH_APP = "launch_app"
    WAIT = "wait"
    BACK = "back"
    HOME = "home"
    PINCH_ZOOM = "pinch_zoom"


@dataclass
class Action:
    """A single atomic action on the device."""

    type: ActionType
    params: dict[str, Any]
    description: str = ""
    is_reversible: bool = True
    risk_level: float = 0.0  # 0.0 = safe, 1.0 = dangerous
    timestamp: float = 0.0  # when action was created
    duration_ms: float = 0.0  # execution time (filled after execution)

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    # ---- Factory methods ----

    @staticmethod
    def tap(x: int, y: int, description: str = "") -> Action:
        return Action(ActionType.TAP, {"x": x, "y": y}, description, is_reversible=True)

    @staticmethod
    def long_press(x: int, y: int, duration_ms: int = 1000) -> Action:
        return Action(
            ActionType.LONG_PRESS,
            {"x": x, "y": y, "duration_ms": duration_ms},
            f"Long press at ({x},{y})",
        )

    @staticmethod
    def double_tap(x: int, y: int) -> Action:
        return Action(ActionType.DOUBLE_TAP, {"x": x, "y": y}, f"Double tap at ({x},{y})")

    @staticmethod
    def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> Action:
        return Action(
            ActionType.SWIPE,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms},
            is_reversible=True,
        )

    @staticmethod
    def type_text(text: str) -> Action:
        return Action(ActionType.TYPE_TEXT, {"text": text}, is_reversible=True)

    @staticmethod
    def back() -> Action:
        return Action(ActionType.BACK, {}, "Press back", is_reversible=True)

    @staticmethod
    def home() -> Action:
        return Action(ActionType.HOME, {}, "Press home", is_reversible=True)

    @staticmethod
    def launch_app(package: str) -> Action:
        return Action(ActionType.LAUNCH_APP, {"package": package}, f"Launch {package}")

    @staticmethod
    def scroll(direction: str = "down", amount: int = 1) -> Action:
        return Action(ActionType.SCROLL, {"direction": direction, "amount": amount})

    @staticmethod
    def wait(ms: int = 1000) -> Action:
        return Action(ActionType.WAIT, {"ms": ms}, f"Wait {ms}ms")

    @staticmethod
    def pinch_zoom(cx: int, cy: int, zoom_in: bool = True, scale: float = 0.5) -> Action:
        return Action(
            ActionType.PINCH_ZOOM,
            {"cx": cx, "cy": cy, "zoom_in": zoom_in, "scale": scale},
            f"Pinch {'in' if zoom_in else 'out'} at ({cx},{cy})",
        )

    # ---- Serialization ----

    def to_dict(self) -> dict[str, Any]:
        """Serialize action to dict for JSON storage."""
        return {
            "type": self.type.value,
            "params": self.params,
            "description": self.description,
            "is_reversible": self.is_reversible,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Deserialize action from dict."""
        return cls(
            type=ActionType(data["type"]),
            params=data.get("params", {}),
            description=data.get("description", ""),
            is_reversible=data.get("is_reversible", True),
            risk_level=data.get("risk_level", 0.0),
            timestamp=data.get("timestamp", 0.0),
            duration_ms=data.get("duration_ms", 0.0),
        )


# ---------------------------------------------------------------------------
# Action Recording (record & replay)
# ---------------------------------------------------------------------------


@dataclass
class ActionRecording:
    """A sequence of recorded actions with timing info."""

    name: str = ""
    actions: list[Action] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def add(self, action: Action) -> None:
        self.actions.append(action)
        self.total_duration_ms += action.duration_ms

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "actions": [a.to_dict() for a in self.actions],
                "total_duration_ms": self.total_duration_ms,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, json_str: str) -> ActionRecording:
        data = json.loads(json_str)
        rec = cls(
            name=data.get("name", ""),
            total_duration_ms=data.get("total_duration_ms", 0.0),
        )
        rec.actions = [Action.from_dict(a) for a in data.get("actions", [])]
        return rec

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())
        logger.info(f"Recording saved: {path} ({len(self.actions)} actions)")

    @classmethod
    def load(cls, path: str | Path) -> ActionRecording:
        content = Path(path).read_text()
        rec = cls.from_json(content)
        logger.info(f"Recording loaded: {path} ({len(rec.actions)} actions)")
        return rec


# ---------------------------------------------------------------------------
# Execution Timer
# ---------------------------------------------------------------------------


class ActionTimer:
    """Context manager for timing action execution."""

    def __init__(self, action: Action) -> None:
        self._action = action
        self._start: float = 0.0

    def __enter__(self) -> ActionTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed = (time.perf_counter() - self._start) * 1000
        self._action.duration_ms = elapsed
        logger.debug(f"Action {self._action.type.value} took {elapsed:.1f}ms")
