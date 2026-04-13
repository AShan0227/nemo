"""Entropy-based routing — System 1 / System 2 decision engine.

Uses Shannon entropy of LLM token logprobs to decide computation depth:
- Low entropy  -> System 1: cached / rule-based fast response
- Mid entropy  -> System 1.5: small model with filtered context
- High entropy -> System 2: full LLM with chain-of-thought reasoning

Reference: "Think Just Enough" (2025) — 25-50% compute savings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ReasoningDepth(Enum):
    SYSTEM_1 = "cached"       # fast, no LLM call
    SYSTEM_1_5 = "lightweight"  # small model or filtered context
    SYSTEM_2 = "full"          # full LLM with CoT


@dataclass
class RoutingDecision:
    depth: ReasoningDepth
    entropy: float
    confidence: float


class EntropyRouter:
    """Route decisions between cached responses and full LLM reasoning."""

    def __init__(self, threshold_low: float = 0.3, threshold_high: float = 0.7) -> None:
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._cache: dict[str, str] = {}  # screen_hash -> action

    def compute_entropy(self, action_probs: list[float]) -> float:
        """Shannon entropy of action probability distribution."""
        entropy = 0.0
        for p in action_probs:
            if p > 0:
                entropy -= p * math.log2(p)
        # Normalize by max possible entropy
        max_entropy = math.log2(max(len(action_probs), 2))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def route(self, screen_hash: str, action_probs: list[float]) -> RoutingDecision:
        """Decide reasoning depth based on screen entropy."""
        entropy = self.compute_entropy(action_probs)
        confidence = 1.0 - entropy

        if entropy < self._threshold_low and screen_hash in self._cache:
            depth = ReasoningDepth.SYSTEM_1
        elif entropy < self._threshold_high:
            depth = ReasoningDepth.SYSTEM_1_5
        else:
            depth = ReasoningDepth.SYSTEM_2

        logger.debug(f"Entropy={entropy:.3f} -> {depth.value} (confidence={confidence:.3f})")
        return RoutingDecision(depth=depth, entropy=entropy, confidence=confidence)

    def cache_action(self, screen_hash: str, action: str) -> None:
        """Cache a successful action for a known screen state."""
        self._cache[screen_hash] = action

    def get_cached_action(self, screen_hash: str) -> str | None:
        return self._cache.get(screen_hash)
