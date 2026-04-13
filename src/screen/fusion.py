"""Multi-modal screen understanding fusion (Dempster-Shafer evidence theory).

Combines signals from:
1. UI Hierarchy (structural)
2. OCR (text recognition)
3. Visual model (screenshot analysis)

Constructive interference = sources agree = high confidence
Destructive interference = sources disagree = low confidence, needs more evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.screen.ocr import OCRTextBlock
    from src.screen.parser import UIElement


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().lower()


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class EvidenceMass:
    """Mass function for Dempster-Shafer fusion."""

    beliefs: dict[str, float]  # hypothesis -> mass

    def __post_init__(self):
        total = sum(self.beliefs.values())
        if total > 0:
            self.beliefs = {k: v / total for k, v in self.beliefs.items()}


def dempster_combine(m1: EvidenceMass, m2: EvidenceMass) -> tuple[EvidenceMass, float]:
    """Combine two independent evidence sources using Dempster's rule.

    Returns:
        (combined_mass, conflict_degree)
    """
    combined: dict[str, float] = {}
    conflict = 0.0

    for h1, v1 in m1.beliefs.items():
        for h2, v2 in m2.beliefs.items():
            product = v1 * v2
            if h1 == h2 or h1 == "unknown" or h2 == "unknown":
                key = h2 if h1 == "unknown" else h1
                combined[key] = combined.get(key, 0.0) + product
            else:
                conflict += product

    normalization = 1.0 - conflict
    if normalization > 0:
        combined = {k: v / normalization for k, v in combined.items()}

    return EvidenceMass(combined), conflict


def fuse_multiple(*masses: EvidenceMass) -> tuple[EvidenceMass, float]:
    """Fuse multiple evidence sources sequentially."""
    if not masses:
        return EvidenceMass({"unknown": 1.0}), 0.0

    result = masses[0]
    total_conflict = 0.0

    for m in masses[1:]:
        result, conflict = dempster_combine(result, m)
        total_conflict = max(total_conflict, conflict)

    return result, total_conflict


@dataclass(frozen=True)
class SignalCandidate:
    """Candidate hypothesis extracted from one modality."""

    label: str
    confidence: float
    source: str
    bounds: tuple[int, int, int, int] | None = None

    def normalized_label(self) -> str:
        return _normalize_label(self.label)


@dataclass
class FusionResult:
    """Final multi-source fusion result."""

    beliefs: dict[str, float]
    ranked: list[tuple[str, float]]
    conflict: float
    source_masses: dict[str, EvidenceMass]

    @property
    def best_label(self) -> str:
        return self.ranked[0][0] if self.ranked else "unknown"


def candidates_from_ui(
    elements: list[UIElement],
    *,
    interactive_only: bool = True,
) -> list[SignalCandidate]:
    """Generate text hypotheses from UI hierarchy."""
    candidates: list[SignalCandidate] = []
    for element in elements:
        if interactive_only and not (element.clickable or element.scrollable or element.editable):
            continue

        if element.text:
            label = element.text
            base_conf = 0.88
        elif element.content_desc:
            label = element.content_desc
            base_conf = 0.82
        elif element.resource_id:
            label = element.resource_id.split("/")[-1]
            base_conf = 0.72
        elif element.class_name:
            label = element.class_name.split(".")[-1]
            base_conf = 0.45
        else:
            continue

        boost = 0.04 if (element.clickable or element.editable) else 0.0
        penalty = 0.08 if element.scrollable else 0.0
        confidence = _clamp(base_conf + boost - penalty, 0.15, 0.99)
        candidates.append(
            SignalCandidate(
                label=label,
                confidence=confidence,
                source="ui",
                bounds=element.bounds,
            )
        )
    return candidates


def candidates_from_ocr(
    blocks: list[OCRTextBlock],
    *,
    min_confidence: float = 0.4,
) -> list[SignalCandidate]:
    """Generate hypotheses from OCR text blocks."""
    candidates: list[SignalCandidate] = []
    for block in blocks:
        if block.confidence < min_confidence:
            continue
        candidates.append(
            SignalCandidate(
                label=block.text,
                confidence=_clamp(block.confidence, 0.0, 1.0),
                source="ocr",
                bounds=block.bounds,
            )
        )
    return candidates


def candidates_from_visual(
    visual_labels: dict[str, float] | list[tuple[str, float]],
) -> list[SignalCandidate]:
    """Generate hypotheses from visual model tags/logits."""
    items = visual_labels.items() if isinstance(visual_labels, dict) else visual_labels
    candidates: list[SignalCandidate] = []
    for label, score in items:
        normalized_score = _clamp(float(score), 0.0, 1.0)
        if normalized_score <= 0:
            continue
        candidates.append(
            SignalCandidate(
                label=label,
                confidence=normalized_score,
                source="visual",
            )
        )
    return candidates


def candidates_to_mass(
    candidates: list[SignalCandidate],
    *,
    min_unknown: float = 0.1,
) -> EvidenceMass:
    """Convert candidates into an evidence mass."""
    if not candidates:
        return EvidenceMass({"unknown": 1.0})

    beliefs: dict[str, float] = {}
    max_conf = 0.0
    for candidate in candidates:
        label = candidate.normalized_label()
        if not label:
            continue
        score = _clamp(candidate.confidence, 0.0, 1.0)
        beliefs[label] = max(beliefs.get(label, 0.0), score)
        max_conf = max(max_conf, score)

    if not beliefs:
        return EvidenceMass({"unknown": 1.0})

    beliefs["unknown"] = max(min_unknown, 1.0 - max_conf)
    return EvidenceMass(beliefs)


def fuse_screen_sources(
    *,
    ui_candidates: list[SignalCandidate] | None = None,
    ocr_candidates: list[SignalCandidate] | None = None,
    visual_candidates: list[SignalCandidate] | None = None,
    top_k: int = 5,
) -> FusionResult:
    """Fuse UI/OCR/Visual candidates into final ranked hypotheses."""
    source_candidates = {
        "ui": ui_candidates or [],
        "ocr": ocr_candidates or [],
        "visual": visual_candidates or [],
    }

    source_masses: dict[str, EvidenceMass] = {}
    masses: list[EvidenceMass] = []
    for source, candidates in source_candidates.items():
        if not candidates:
            continue
        mass = candidates_to_mass(candidates)
        source_masses[source] = mass
        masses.append(mass)

    if not masses:
        empty = {"unknown": 1.0}
        return FusionResult(
            beliefs=empty,
            ranked=[("unknown", 1.0)],
            conflict=0.0,
            source_masses={},
        )

    combined, conflict = fuse_multiple(*masses)
    ranked = sorted(
        combined.beliefs.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if top_k > 0:
        ranked = ranked[:top_k]
    return FusionResult(
        beliefs=combined.beliefs,
        ranked=ranked,
        conflict=conflict,
        source_masses=source_masses,
    )
