"""Intent tracker — simplified particle filter for ambiguous commands.

When a user command is ambiguous (e.g., "open settings" could mean
system settings or app-specific settings), maintains multiple
intent hypotheses and collapses them as evidence accumulates.

Inspired by quantum superposition — multiple intents exist simultaneously
until "measured" (confirmed by screen state evidence).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class IntentHypothesis:
    """A candidate interpretation of the user's command."""
    description: str
    weight: float = 1.0
    evidence_for: int = 0
    evidence_against: int = 0

    @property
    def normalized_weight(self) -> float:
        """Weight is always recalculated during resampling."""
        return self.weight


@dataclass
class IntentState:
    """Current state of intent resolution."""
    collapsed: bool
    dominant: Optional[IntentHypothesis]
    hypotheses: List[IntentHypothesis]
    entropy: float  # 0 = certain, 1 = maximum uncertainty


class IntentTracker:
    """Track and resolve ambiguous user intents via particle filter.

    Usage:
        tracker = IntentTracker(["open system settings", "open app settings"])
        while not tracker.is_collapsed:
            screen = observe()
            tracker.update_evidence(screen_text)
            intent = tracker.get_best_intent()
    """

    def __init__(
        self,
        hypotheses: List[str],
        collapse_threshold: float = 0.8,
        max_steps_before_ask: int = 5,
    ) -> None:
        self._hypotheses = [
            IntentHypothesis(description=h, weight=1.0 / len(hypotheses))
            for h in hypotheses
        ]
        self._collapse_threshold = collapse_threshold
        self._max_steps = max_steps_before_ask
        self._step_count = 0

    @property
    def is_collapsed(self) -> bool:
        """True if one hypothesis dominates."""
        if not self._hypotheses:
            return True
        best = max(self._hypotheses, key=lambda h: h.weight)
        return best.weight >= self._collapse_threshold

    @property
    def should_ask_user(self) -> bool:
        """True if too many steps without collapse — need clarification."""
        return self._step_count >= self._max_steps and not self.is_collapsed

    def update_evidence(self, screen_text: str) -> None:
        """Update hypothesis weights based on observed screen text."""
        self._step_count += 1

        for h in self._hypotheses:
            keywords = set(h.description.lower().split())
            screen_lower = screen_text.lower()
            matches = sum(1 for kw in keywords if kw in screen_lower)
            total_kw = len(keywords)

            if total_kw > 0:
                match_ratio = matches / total_kw
                if match_ratio > 0.3:
                    h.weight *= 1.0 + match_ratio
                    h.evidence_for += 1
                elif match_ratio == 0:
                    h.weight *= 0.7
                    h.evidence_against += 1

        # Normalize weights
        self._normalize()

        # Log state
        state = self.get_state()
        logger.debug(
            f"Intent tracker step {self._step_count}: "
            f"entropy={state.entropy:.3f}, "
            f"collapsed={state.collapsed}"
        )

    def _normalize(self) -> None:
        """Normalize hypothesis weights to sum to 1."""
        total = sum(h.weight for h in self._hypotheses)
        if total > 0:
            for h in self._hypotheses:
                h.weight /= total

    def get_best_intent(self) -> str:
        """Return the most likely intent description."""
        if not self._hypotheses:
            return ""
        best = max(self._hypotheses, key=lambda h: h.weight)
        return best.description

    def get_state(self) -> IntentState:
        """Get current intent resolution state."""
        import math

        if not self._hypotheses:
            return IntentState(collapsed=True, dominant=None, hypotheses=[], entropy=0.0)

        # Shannon entropy of weights
        entropy = 0.0
        for h in self._hypotheses:
            if h.weight > 0:
                entropy -= h.weight * math.log2(h.weight)
        max_entropy = math.log2(max(len(self._hypotheses), 2))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        best = max(self._hypotheses, key=lambda h: h.weight)
        return IntentState(
            collapsed=self.is_collapsed,
            dominant=best if self.is_collapsed else None,
            hypotheses=list(self._hypotheses),
            entropy=normalized_entropy,
        )
