"""Immune System — Negative Selection Algorithm (NSA) for anomaly detection.

Learns "normal" screen transition patterns (self set) and generates detectors
that match only non-self patterns. When a detector fires, it indicates the
agent has entered an unexpected state (crash dialog, wrong app, captcha).

Based on: Forrest et al., 1994 — "Self-Nonself Discrimination in a Computer"

Feature vector for a screen state:
  [element_count, interactive_count, text_length, has_input, has_button,
   has_scrollable, unique_classes, depth_estimate]
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger


@dataclass
class ScreenFeatures:
    """Feature vector extracted from a screen state."""
    element_count: int
    interactive_count: int
    text_length: int
    has_input: bool
    has_button: bool
    has_scrollable: bool
    unique_classes: int
    depth_estimate: int  # max nesting depth

    def to_vector(self) -> List[float]:
        return [
            self.element_count / 100.0,
            self.interactive_count / 50.0,
            self.text_length / 500.0,
            1.0 if self.has_input else 0.0,
            1.0 if self.has_button else 0.0,
            1.0 if self.has_scrollable else 0.0,
            self.unique_classes / 20.0,
            self.depth_estimate / 10.0,
        ]


@dataclass
class Detector:
    """Anomaly detector — a hypersphere in feature space."""
    center: List[float]
    radius: float

    def matches(self, features: List[float]) -> bool:
        """True if the feature vector falls within this detector."""
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(self.center, features)))
        return dist <= self.radius


@dataclass
class AnomalyResult:
    """Result of anomaly check."""
    is_anomaly: bool
    matching_detectors: int
    confidence: float  # 0-1, how anomalous
    message: str = ""


class ImmuneSystem:
    """NSA-based anomaly detection for screen states.

    Training phase: collect self (normal) patterns.
    Detection phase: generate detectors that don't match self,
    then use them to flag non-self (anomalous) patterns.
    """

    def __init__(
        self,
        detector_count: int = 50,
        detector_radius: float = 0.3,
        self_radius: float = 0.2,
        anomaly_threshold: int = 3,
    ) -> None:
        self._self_set: List[List[float]] = []
        self._detectors: List[Detector] = []
        self._detector_count = detector_count
        self._detector_radius = detector_radius
        self._self_radius = self_radius
        self._anomaly_threshold = anomaly_threshold
        self._trained = False

    def add_self_sample(self, features: ScreenFeatures) -> None:
        """Add a normal (self) pattern during training."""
        self._self_set.append(features.to_vector())
        self._trained = False  # invalidate detectors

    def train(self, max_candidates: int = 1000) -> int:
        """Generate detectors via negative selection.

        Returns number of detectors generated.
        """
        if not self._self_set:
            logger.warning("No self samples to train on")
            return 0

        dim = len(self._self_set[0])
        self._detectors = []

        candidates_tried = 0
        while len(self._detectors) < self._detector_count and candidates_tried < max_candidates:
            # Generate random candidate detector
            center = [random.uniform(0, 1.5) for _ in range(dim)]
            candidates_tried += 1

            # Check if it matches any self pattern
            matches_self = False
            for self_vec in self._self_set:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(center, self_vec)))
                if dist < self._self_radius:
                    matches_self = True
                    break

            if not matches_self:
                self._detectors.append(Detector(center, self._detector_radius))

        self._trained = True
        logger.info(
            f"NSA trained: {len(self._detectors)} detectors from "
            f"{len(self._self_set)} self samples ({candidates_tried} candidates)"
        )
        return len(self._detectors)

    def check(self, features: ScreenFeatures) -> AnomalyResult:
        """Check if a screen state is anomalous (non-self)."""
        if not self._trained:
            return AnomalyResult(
                is_anomaly=False, matching_detectors=0, confidence=0.0,
                message="Not trained yet",
            )

        vec = features.to_vector()
        matching = sum(1 for d in self._detectors if d.matches(vec))
        confidence = min(matching / max(self._anomaly_threshold, 1), 1.0)

        is_anomaly = matching >= self._anomaly_threshold
        if is_anomaly:
            logger.warning(
                f"Anomaly detected: {matching}/{len(self._detectors)} detectors fired "
                f"(threshold={self._anomaly_threshold})"
            )

        return AnomalyResult(
            is_anomaly=is_anomaly,
            matching_detectors=matching,
            confidence=confidence,
            message=f"{matching} detectors fired" if is_anomaly else "Normal",
        )


def extract_features(screen_state) -> ScreenFeatures:
    """Extract feature vector from a ScreenState object."""
    elements = screen_state.elements
    interactive = screen_state.interactive_elements

    classes = set()
    max_depth = 0
    total_text_len = 0
    has_input = False
    has_button = False
    has_scrollable = False

    for e in elements:
        classes.add(e.class_name)
        total_text_len += len(e.display_text)
        if e.editable:
            has_input = True
        if e.clickable and "Button" in e.class_name:
            has_button = True
        if e.scrollable:
            has_scrollable = True

    return ScreenFeatures(
        element_count=len(elements),
        interactive_count=len(interactive),
        text_length=total_text_len,
        has_input=has_input,
        has_button=has_button,
        has_scrollable=has_scrollable,
        unique_classes=len(classes),
        depth_estimate=max_depth,
    )
