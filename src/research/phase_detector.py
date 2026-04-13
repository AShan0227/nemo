"""Phase transition detector — detect capability jumps (grokking).

Tracks agent performance metrics over time and detects sudden
improvements, analogous to phase transitions in physics.

When a grokking event is detected, it indicates the agent has
crossed a capability threshold (e.g., learned a new app pattern).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class PerformanceSnapshot:
    """Metrics at a point in time."""
    timestamp: float  # epoch seconds
    success_rate: float  # 0-1
    avg_steps: float
    graph_nodes: int
    graph_edges: int
    task_count: int


@dataclass
class PhaseTransition:
    """Detected capability jump."""
    timestamp: float
    metric_name: str
    old_value: float
    new_value: float
    change_ratio: float


class PhaseDetector:
    """Detect sudden capability improvements via sliding-window variance analysis.

    Monitors rolling metrics and triggers when the recent window's mean
    deviates significantly from the historical baseline.
    """

    def __init__(
        self,
        window_size: int = 10,
        baseline_size: int = 50,
        sensitivity: float = 2.0,
    ) -> None:
        """
        Args:
            window_size: Recent window for detecting change
            baseline_size: Historical baseline window
            sensitivity: Number of std devs for transition detection
        """
        self._window_size = window_size
        self._baseline_size = baseline_size
        self._sensitivity = sensitivity
        self._history: dict[str, deque] = {
            "success_rate": deque(maxlen=baseline_size),
            "avg_steps": deque(maxlen=baseline_size),
            "graph_coverage": deque(maxlen=baseline_size),
        }
        self._transitions: list[PhaseTransition] = []

    def record(self, snapshot: PerformanceSnapshot) -> Optional[PhaseTransition]:
        """Record a new performance snapshot and check for phase transitions.

        Returns PhaseTransition if one is detected, None otherwise.
        """
        self._history["success_rate"].append(snapshot.success_rate)
        self._history["avg_steps"].append(snapshot.avg_steps)
        coverage = snapshot.graph_nodes + snapshot.graph_edges
        self._history["graph_coverage"].append(coverage)

        # Need enough history for meaningful detection
        if len(self._history["success_rate"]) < self._window_size + 5:
            return None

        # Check each metric for phase transition
        for metric_name, values in self._history.items():
            transition = self._check_transition(
                metric_name, list(values), snapshot.timestamp
            )
            if transition:
                self._transitions.append(transition)
                logger.info(
                    f"Phase transition detected: {metric_name} "
                    f"{transition.old_value:.3f} -> {transition.new_value:.3f} "
                    f"(change={transition.change_ratio:.2f}x)"
                )
                return transition

        return None

    def _check_transition(
        self, metric_name: str, values: list[float], timestamp: float
    ) -> Optional[PhaseTransition]:
        """Check if recent window shows a significant deviation from baseline."""
        if len(values) < self._window_size + 5:
            return None

        recent = values[-self._window_size:]
        baseline = values[:-self._window_size]

        if not baseline:
            return None

        baseline_mean = sum(baseline) / len(baseline)
        recent_mean = sum(recent) / len(recent)

        # Compute baseline standard deviation
        variance = sum((x - baseline_mean) ** 2 for x in baseline) / len(baseline)
        std = variance ** 0.5

        if std < 0.001:
            # No variance in baseline — check if recent differs at all
            if abs(recent_mean - baseline_mean) > 0.05:
                change = recent_mean / max(baseline_mean, 0.001)
                return PhaseTransition(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    old_value=baseline_mean,
                    new_value=recent_mean,
                    change_ratio=change,
                )
            return None

        # Z-score of recent window mean
        z_score = abs(recent_mean - baseline_mean) / std

        # For avg_steps, improvement means DECREASE (fewer steps)
        is_improvement = (
            recent_mean > baseline_mean if metric_name != "avg_steps"
            else recent_mean < baseline_mean
        )

        if z_score > self._sensitivity and is_improvement:
            change = recent_mean / max(baseline_mean, 0.001)
            return PhaseTransition(
                timestamp=timestamp,
                metric_name=metric_name,
                old_value=baseline_mean,
                new_value=recent_mean,
                change_ratio=change,
            )

        return None

    @property
    def transitions(self) -> list[PhaseTransition]:
        return list(self._transitions)
