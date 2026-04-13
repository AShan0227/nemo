"""High-level action primitives built on ADB controller."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "params": self.params,
            "description": self.description,
            "is_reversible": self.is_reversible,
            "risk_level": self.risk_level,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            type=ActionType(str(data.get("type", ActionType.WAIT.value))),
            params=dict(data.get("params", {})),
            description=str(data.get("description", "")),
            is_reversible=bool(data.get("is_reversible", True)),
            risk_level=float(data.get("risk_level", 0.0)),
        )


@dataclass
class ActionRecord:
    """Single action execution event for recording/replay."""

    timestamp_ms: int
    action: Action
    success: bool
    error: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "action": self.action.to_dict(),
            "success": self.success,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionRecord:
        return cls(
            timestamp_ms=int(data.get("timestamp_ms", 0)),
            action=Action.from_dict(dict(data.get("action", {}))),
            success=bool(data.get("success", False)),
            error=str(data.get("error", "")),
            duration_ms=float(data.get("duration_ms", 0.0)),
        )


class ActionRecorder:
    """Persist and load action execution traces."""

    def __init__(self) -> None:
        self._records: list[ActionRecord] = []

    @property
    def records(self) -> list[ActionRecord]:
        return list(self._records)

    def record(
        self,
        *,
        timestamp_ms: int,
        action: Action,
        success: bool,
        error: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        self._records.append(
            ActionRecord(
                timestamp_ms=timestamp_ms,
                action=action,
                success=success,
                error=error,
                duration_ms=duration_ms,
            )
        )

    def clear(self) -> None:
        self._records.clear()

    def to_dict(self) -> dict[str, Any]:
        return {"records": [record.to_dict() for record in self._records]}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ActionRecorder:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        recorder = cls()
        for record_data in data.get("records", []):
            recorder._records.append(ActionRecord.from_dict(record_data))
        return recorder


@dataclass
class ActionTimingSummary:
    """Aggregate execution timing per action type."""

    count: int = 0
    success_count: int = 0
    total_duration_ms: float = 0.0

    @property
    def avg_duration_ms(self) -> float:
        return self.total_duration_ms / self.count if self.count > 0 else 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0


class ActionTimingTracker:
    """Collect duration/success stats for executed actions."""

    def __init__(self) -> None:
        self._stats: dict[ActionType, ActionTimingSummary] = defaultdict(ActionTimingSummary)

    def record(self, action_type: ActionType, duration_ms: float, success: bool) -> None:
        summary = self._stats[action_type]
        summary.count += 1
        summary.total_duration_ms += max(0.0, duration_ms)
        if success:
            summary.success_count += 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for action_type, summary in self._stats.items():
            result[action_type.value] = {
                "count": float(summary.count),
                "success_count": float(summary.success_count),
                "success_rate": summary.success_rate,
                "avg_duration_ms": summary.avg_duration_ms,
            }
        return result
