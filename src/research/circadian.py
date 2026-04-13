"""Circadian Rhythm modeling — time-aware agent behavior.

Adapts agent behavior based on time of day, day of week, and
usage patterns. Enables proactive anticipation of user needs.

For example:
- Morning: quick actions (check messages, weather)
- Work hours: productivity apps
- Evening: entertainment, browsing
- Night: reduced activity, quiet mode
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class TimeSlot:
    """Activity pattern for a specific time slot."""
    hour: int  # 0-23
    day_of_week: int  # 0=Monday, 6=Sunday
    app_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_tasks: int = 0
    avg_task_duration_ms: float = 0.0

    @property
    def top_apps(self) -> List[Tuple[str, int]]:
        """Most frequently used apps in this time slot."""
        return sorted(self.app_counts.items(), key=lambda x: x[1], reverse=True)[:5]


class CircadianModel:
    """Model and predict user behavior patterns over time.

    Maintains a 24x7 grid of time slots (hourly x day-of-week)
    and tracks what apps/actions the user performs at each slot.
    """

    def __init__(self) -> None:
        # 24 hours x 7 days
        self._slots: Dict[Tuple[int, int], TimeSlot] = {}
        self._task_history: List[Dict] = []

    def _get_slot(self, hour: int, dow: int) -> TimeSlot:
        key = (hour, dow)
        if key not in self._slots:
            self._slots[key] = TimeSlot(hour=hour, day_of_week=dow)
        return self._slots[key]

    def record_activity(
        self,
        app_package: str,
        action_type: str,
        duration_ms: float = 0,
        timestamp: Optional[float] = None,
    ) -> None:
        """Record an activity at the current (or specified) time."""
        dt = datetime.fromtimestamp(timestamp or time.time())
        slot = self._get_slot(dt.hour, dt.weekday())

        slot.app_counts[app_package] += 1
        slot.action_counts[action_type] += 1
        slot.total_tasks += 1

        if duration_ms > 0:
            # EMA of task duration
            if slot.avg_task_duration_ms == 0:
                slot.avg_task_duration_ms = duration_ms
            else:
                slot.avg_task_duration_ms = (
                    0.8 * slot.avg_task_duration_ms + 0.2 * duration_ms
                )

    def predict_apps(
        self, timestamp: Optional[float] = None, top_k: int = 3
    ) -> List[str]:
        """Predict which apps the user is likely to use right now."""
        dt = datetime.fromtimestamp(timestamp or time.time())
        slot = self._get_slot(dt.hour, dt.weekday())

        if not slot.app_counts:
            # Fall back to same hour, any day
            combined: Dict[str, int] = defaultdict(int)
            for (h, _), s in self._slots.items():
                if h == dt.hour:
                    for app, count in s.app_counts.items():
                        combined[app] += count
            if combined:
                return [app for app, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]]
            return []

        return [app for app, _ in slot.top_apps[:top_k]]

    def get_behavior_modifier(self, timestamp: Optional[float] = None) -> Dict[str, float]:
        """Get behavioral modifiers for the current time.

        Returns adjustments to agent parameters based on time patterns.
        """
        dt = datetime.fromtimestamp(timestamp or time.time())
        hour = dt.hour

        modifiers = {
            "action_delay_multiplier": 1.0,
            "verification_level": 1.0,
            "proactive_suggestions": 0.0,
        }

        # Time-based defaults
        if 0 <= hour < 7:
            # Night/early morning: slow down, minimize actions
            modifiers["action_delay_multiplier"] = 1.5
            modifiers["verification_level"] = 0.5
            modifiers["proactive_suggestions"] = 0.0
        elif 7 <= hour < 9:
            # Morning routine: quick actions, predictable
            modifiers["action_delay_multiplier"] = 0.8
            modifiers["proactive_suggestions"] = 0.8
        elif 9 <= hour < 18:
            # Work hours: normal speed, high verification
            modifiers["action_delay_multiplier"] = 1.0
            modifiers["verification_level"] = 1.0
            modifiers["proactive_suggestions"] = 0.5
        elif 18 <= hour < 22:
            # Evening: relaxed, moderate suggestions
            modifiers["action_delay_multiplier"] = 1.1
            modifiers["verification_level"] = 0.8
            modifiers["proactive_suggestions"] = 0.6
        else:
            # Late night: quiet
            modifiers["action_delay_multiplier"] = 1.3
            modifiers["proactive_suggestions"] = 0.2

        return modifiers

    @property
    def coverage(self) -> Dict[str, int]:
        """How much data has been collected."""
        return {
            "total_slots_observed": len(self._slots),
            "total_slots_possible": 24 * 7,
            "total_activities": sum(s.total_tasks for s in self._slots.values()),
        }
