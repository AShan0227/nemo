"""High-level action primitives built on ADB controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(Enum):
    TAP = "tap"
    LONG_PRESS = "long_press"
    DOUBLE_TAP = "double_tap"
    PINCH_ZOOM = "pinch_zoom"
    SWIPE = "swipe"
    TYPE_TEXT = "type_text"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    LAUNCH_APP = "launch_app"
    WAIT = "wait"
    BACK = "back"
    HOME = "home"


@dataclass
class Action:
    """A single atomic action on the device."""

    type: ActionType
    params: dict[str, Any]
    description: str = ""
    is_reversible: bool = True
    risk_level: float = 0.0  # 0.0 = safe, 1.0 = dangerous

    @staticmethod
    def tap(x: int, y: int, description: str = "") -> Action:
        return Action(ActionType.TAP, {"x": x, "y": y}, description, is_reversible=True)

    @staticmethod
    def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> Action:
        return Action(
            ActionType.SWIPE,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "duration_ms": duration_ms},
            is_reversible=True,
        )

    @staticmethod
    def long_press(x: int, y: int, duration_ms: int = 800) -> Action:
        return Action(
            ActionType.LONG_PRESS,
            {"x": x, "y": y, "duration_ms": duration_ms},
            is_reversible=True,
        )

    @staticmethod
    def double_tap(x: int, y: int, interval_ms: int = 120) -> Action:
        return Action(
            ActionType.DOUBLE_TAP,
            {"x": x, "y": y, "interval_ms": interval_ms},
            is_reversible=True,
        )

    @staticmethod
    def pinch_zoom(
        x: int,
        y: int,
        *,
        distance: int = 220,
        zoom_in: bool = True,
        duration_ms: int = 350,
    ) -> Action:
        return Action(
            ActionType.PINCH_ZOOM,
            {
                "x": x,
                "y": y,
                "distance": distance,
                "zoom_in": zoom_in,
                "duration_ms": duration_ms,
            },
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
