"""Entropy-based routing — System 1 / System 1.5 / System 2 decision engine.

Uses Shannon entropy of action probability distributions to decide computation depth:
- Low entropy  -> System 1: cached / rule-based fast response (no LLM call)
- Mid entropy  -> System 1.5: small model with filtered context (shorter prompt)
- High entropy -> System 2: full LLM with chain-of-thought reasoning

Reference: "Think Just Enough" (2025) — 25-50% compute savings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


class ReasoningDepth(Enum):
    SYSTEM_1 = "cached"         # fast, no LLM call
    SYSTEM_1_5 = "lightweight"  # small model or filtered context
    SYSTEM_2 = "full"           # full LLM with CoT


@dataclass
class RoutingDecision:
    depth: ReasoningDepth
    entropy: float
    confidence: float


@dataclass
class RouterStats:
    """Track routing statistics for monitoring."""
    system1_hits: int = 0
    system1_5_hits: int = 0
    system2_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def total(self) -> int:
        return self.system1_hits + self.system1_5_hits + self.system2_hits

    @property
    def savings_ratio(self) -> float:
        """Fraction of decisions that avoided full LLM call."""
        if self.total == 0:
            return 0.0
        return (self.system1_hits + self.system1_5_hits) / self.total


class EntropyRouter:
    """Route decisions between cached responses and full LLM reasoning."""

    def __init__(self, threshold_low: float = 0.3, threshold_high: float = 0.7) -> None:
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._cache: dict[str, str] = {}  # screen_hash -> action JSON
        self.stats = RouterStats()

    def compute_entropy(self, action_probs: list[float]) -> float:
        """Shannon entropy of action probability distribution (normalized to [0,1])."""
        entropy = 0.0
        for p in action_probs:
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(max(len(action_probs), 2))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def route(self, screen_hash: str, action_probs: list[float]) -> RoutingDecision:
        """Decide reasoning depth based on screen entropy."""
        entropy = self.compute_entropy(action_probs)
        confidence = 1.0 - entropy

        if entropy < self._threshold_low and screen_hash in self._cache:
            depth = ReasoningDepth.SYSTEM_1
            self.stats.system1_hits += 1
            self.stats.cache_hits += 1
        elif entropy < self._threshold_high:
            depth = ReasoningDepth.SYSTEM_1_5
            self.stats.system1_5_hits += 1
            if screen_hash not in self._cache:
                self.stats.cache_misses += 1
        else:
            depth = ReasoningDepth.SYSTEM_2
            self.stats.system2_hits += 1

        logger.debug(f"Entropy={entropy:.3f} -> {depth.value} (confidence={confidence:.3f})")
        return RoutingDecision(depth=depth, entropy=entropy, confidence=confidence)

    def cache_action(self, screen_hash: str, action: str) -> None:
        """Cache a successful action for a known screen state."""
        self._cache[screen_hash] = action

    def get_cached_action(self, screen_hash: str) -> Optional[str]:
        return self._cache.get(screen_hash)

    def filter_elements_by_relevance(
        self, elements: list, task: str, top_k: int = 10
    ) -> list:
        """For System 1.5: filter to top-k most relevant elements.

        Simple heuristic: prefer elements whose text overlaps with task keywords.
        """
        if len(elements) <= top_k:
            return elements

        task_words = set(task.lower().split())

        def relevance_score(elem) -> float:
            text = elem.display_text.lower()
            return sum(1 for w in task_words if w in text)

        scored = sorted(elements, key=relevance_score, reverse=True)
        return scored[:top_k]
