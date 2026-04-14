"""Entropy-based routing for System 1 / 1.5 / 2 decision depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from loguru import logger


class ReasoningDepth(Enum):
    SYSTEM_1 = "cached"         # fast, no LLM call
    SYSTEM_1_5 = "lightweight"  # reduced compute
    SYSTEM_2 = "full"           # full LLM reasoning


@dataclass
class RoutingDecision:
    depth: ReasoningDepth
    entropy: float
    confidence: float
    source: str = "unknown"


@dataclass
class RouterStats:
    """Track routing statistics for monitoring."""

    system1_hits: int = 0
    system1_5_hits: int = 0
    system2_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    real_entropy_routes: int = 0
    heuristic_routes: int = 0

    @property
    def total(self) -> int:
        return self.system1_hits + self.system1_5_hits + self.system2_hits

    @property
    def savings_ratio(self) -> float:
        """Fraction of decisions that avoided full LLM call."""
        if self.total == 0:
            return 0.0
        return (self.system1_hits + self.system1_5_hits) / self.total


@dataclass
class RoutingBenchmarkCase:
    """Single benchmark sample for comparing heuristic vs real entropy routing."""

    screen_hash: str
    expected_depth: ReasoningDepth
    action_probs: list[float]
    real_entropy: float
    cached: bool = False


@dataclass
class RoutingAccuracyReport:
    sample_size: int
    heuristic_accuracy: float
    real_entropy_accuracy: float

    @property
    def delta(self) -> float:
        return self.real_entropy_accuracy - self.heuristic_accuracy


class EntropyRouter:
    """Route decisions between cached responses and full LLM reasoning."""

    def __init__(self, threshold_low: float = 0.3, threshold_high: float = 0.7) -> None:
        self._threshold_low = threshold_low
        self._threshold_high = threshold_high
        self._cache: dict[str, str] = {}  # screen_hash -> action JSON
        self._entropy_memory: dict[str, float] = {}  # screen_hash -> observed entropy
        self.stats = RouterStats()

    def compute_entropy(self, action_probs: list[float]) -> float:
        """Shannon entropy of action probability distribution (normalized to [0,1])."""
        entropy = 0.0
        for p in action_probs:
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(max(len(action_probs), 2))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def observe_entropy(self, screen_hash: str, entropy: float) -> None:
        """Store real entropy observed from model logprobs for future routing."""
        if not math.isfinite(entropy):
            return
        self._entropy_memory[screen_hash] = min(max(float(entropy), 0.0), 1.0)

    def get_observed_entropy(self, screen_hash: str) -> float | None:
        return self._entropy_memory.get(screen_hash)

    def route(
        self,
        screen_hash: str,
        action_probs: list[float] | None = None,
        *,
        entropy: float | None = None,
    ) -> RoutingDecision:
        """Decide reasoning depth.

        Priority:
        1) Explicit real entropy input
        2) Previously observed entropy for this screen
        3) Heuristic action probability entropy
        4) Cold start fallback (high entropy)
        """
        source = "cold_start"
        resolved_entropy = entropy

        if resolved_entropy is not None:
            source = "real_entropy"
            self.stats.real_entropy_routes += 1
        else:
            observed = self._entropy_memory.get(screen_hash)
            if observed is not None:
                resolved_entropy = observed
                source = "observed_entropy"
                self.stats.real_entropy_routes += 1
            elif action_probs is not None:
                resolved_entropy = self.compute_entropy(action_probs)
                source = "heuristic_probs"
                self.stats.heuristic_routes += 1
            else:
                resolved_entropy = 1.0
                self.stats.heuristic_routes += 1

        resolved_entropy = min(max(float(resolved_entropy), 0.0), 1.0)
        confidence = 1.0 - resolved_entropy
        depth = self._depth_from_entropy(screen_hash, resolved_entropy)

        logger.debug(
            "Entropy={:.3f} source={} -> {} (confidence={:.3f})",
            resolved_entropy,
            source,
            depth.value,
            confidence,
        )
        return RoutingDecision(
            depth=depth,
            entropy=resolved_entropy,
            confidence=confidence,
            source=source,
        )

    def _depth_from_entropy(self, screen_hash: str, entropy: float) -> ReasoningDepth:
        if entropy < self._threshold_low and screen_hash in self._cache:
            self.stats.system1_hits += 1
            self.stats.cache_hits += 1
            return ReasoningDepth.SYSTEM_1
        if entropy < self._threshold_high:
            self.stats.system1_5_hits += 1
            if screen_hash not in self._cache:
                self.stats.cache_misses += 1
            return ReasoningDepth.SYSTEM_1_5
        self.stats.system2_hits += 1
        return ReasoningDepth.SYSTEM_2

    def cache_action(self, screen_hash: str, action: str) -> None:
        """Cache a successful action for a known screen state."""
        self._cache[screen_hash] = action

    def get_cached_action(self, screen_hash: str) -> str | None:
        return self._cache.get(screen_hash)

    def filter_elements_by_relevance(
        self,
        elements: list,
        task: str,
        top_k: int = 10,
    ) -> list:
        """For System 1.5: filter to top-k most relevant elements."""
        if len(elements) <= top_k:
            return elements

        task_words = set(task.lower().split())

        def relevance_score(elem: object) -> float:
            text = getattr(elem, "display_text", "").lower()
            return float(sum(1 for w in task_words if w in text))

        scored = sorted(elements, key=relevance_score, reverse=True)
        return scored[:top_k]


def compare_routing_accuracy(
    cases: list[RoutingBenchmarkCase],
    threshold_low: float = 0.3,
    threshold_high: float = 0.7,
) -> RoutingAccuracyReport:
    """Compare heuristic routing vs real-entropy routing accuracy on labeled cases."""
    if not cases:
        return RoutingAccuracyReport(
            sample_size=0,
            heuristic_accuracy=0.0,
            real_entropy_accuracy=0.0,
        )

    heuristic_router = EntropyRouter(threshold_low, threshold_high)
    real_router = EntropyRouter(threshold_low, threshold_high)

    heuristic_hits = 0
    real_hits = 0

    for case in cases:
        if case.cached:
            heuristic_router.cache_action(case.screen_hash, '{"action":"tap"}')
            real_router.cache_action(case.screen_hash, '{"action":"tap"}')

        h_depth = heuristic_router.route(case.screen_hash, action_probs=case.action_probs).depth
        r_depth = real_router.route(case.screen_hash, entropy=case.real_entropy).depth

        if h_depth == case.expected_depth:
            heuristic_hits += 1
        if r_depth == case.expected_depth:
            real_hits += 1

    size = len(cases)
    return RoutingAccuracyReport(
        sample_size=size,
        heuristic_accuracy=heuristic_hits / size,
        real_entropy_accuracy=real_hits / size,
    )
