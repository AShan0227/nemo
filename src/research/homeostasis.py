"""Homeostatic regulation — self-correcting behavior via negative feedback.

Maintains system health variables at setpoints, analogous to biological
homeostasis (body temperature, blood sugar, etc.).

Each regulated variable has:
- A setpoint (target value)
- A sensor (measures current value)
- A drive (error = setpoint - current)
- An effector (adjustment signal to modify behavior)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

from loguru import logger


@dataclass
class RegulatedVariable:
    """A single homeostatic variable."""
    name: str
    setpoint: float
    current: float = 0.0
    gain: float = 1.0  # how aggressively to correct
    history: deque = field(default_factory=lambda: deque(maxlen=50))

    @property
    def drive(self) -> float:
        """Error signal: how far from setpoint."""
        return self.setpoint - self.current

    @property
    def normalized_drive(self) -> float:
        """Drive normalized to [-1, 1] range."""
        if self.setpoint == 0:
            return 0.0
        return max(-1.0, min(1.0, self.drive / self.setpoint))

    def update(self, value: float) -> None:
        """Update current value from sensor."""
        self.current = value
        self.history.append(value)

    @property
    def trend(self) -> float:
        """Recent trend: positive = improving, negative = degrading."""
        if len(self.history) < 3:
            return 0.0
        recent = list(self.history)[-3:]
        return (recent[-1] - recent[0]) / 2.0


@dataclass
class Adjustment:
    """Behavioral adjustment output from homeostatic regulator."""
    name: str
    description: str
    value: float  # adjustment magnitude


class HomeostasisRegulator:
    """Maintains agent health via negative feedback loops.

    Monitors success_rate, response_latency, and error_rate.
    Outputs adjustment signals that modify agent behavior.
    """

    def __init__(self) -> None:
        self.variables: Dict[str, RegulatedVariable] = {
            "success_rate": RegulatedVariable(
                name="success_rate",
                setpoint=0.9,
                gain=1.0,
            ),
            "response_latency_ms": RegulatedVariable(
                name="response_latency_ms",
                setpoint=2000.0,
                gain=0.5,
            ),
            "error_rate": RegulatedVariable(
                name="error_rate",
                setpoint=0.05,
                gain=1.5,
            ),
        }

    def update_metrics(
        self,
        success_rate: float = None,
        response_latency_ms: float = None,
        error_rate: float = None,
    ) -> None:
        """Update regulated variables from measured metrics."""
        if success_rate is not None:
            self.variables["success_rate"].update(success_rate)
        if response_latency_ms is not None:
            self.variables["response_latency_ms"].update(response_latency_ms)
        if error_rate is not None:
            self.variables["error_rate"].update(error_rate)

    def get_adjustments(self) -> List[Adjustment]:
        """Compute behavioral adjustments based on current drives."""
        adjustments = []

        sr = self.variables["success_rate"]
        if sr.drive > 0.1:
            # Success rate too low — increase verification
            adjustments.append(Adjustment(
                name="verify_actions",
                description="Add extra observe step after each action",
                value=sr.drive * sr.gain,
            ))

        latency = self.variables["response_latency_ms"]
        if latency.drive < -500:
            # Latency too high — reduce prompt size
            adjustments.append(Adjustment(
                name="reduce_prompt",
                description="Filter to top-k elements in prompt",
                value=abs(latency.normalized_drive) * latency.gain,
            ))

        er = self.variables["error_rate"]
        if er.drive < -0.05:
            # Error rate too high — slow down
            adjustments.append(Adjustment(
                name="increase_delay",
                description="Increase action delay between steps",
                value=abs(er.drive) * er.gain,
            ))

        if adjustments:
            logger.debug(
                f"Homeostasis: {len(adjustments)} adjustments — "
                + ", ".join(f"{a.name}={a.value:.3f}" for a in adjustments)
            )

        return adjustments

    @property
    def health_summary(self) -> Dict[str, Dict[str, float]]:
        """Current health status of all regulated variables."""
        return {
            name: {
                "current": var.current,
                "setpoint": var.setpoint,
                "drive": var.drive,
                "trend": var.trend,
            }
            for name, var in self.variables.items()
        }
