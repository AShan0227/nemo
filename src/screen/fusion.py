"""Multi-modal screen understanding fusion (Dempster-Shafer evidence theory).

Combines signals from:
1. UI Hierarchy (structural)
2. OCR (text recognition)
3. Visual model (screenshot analysis)

Constructive interference = sources agree = high confidence
Destructive interference = sources disagree = low confidence, needs more evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


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
