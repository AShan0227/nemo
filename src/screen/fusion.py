"""Multi-modal screen understanding fusion (Dempster-Shafer evidence theory).

Combines signals from:
1. UI Hierarchy (structural)
2. OCR (text recognition)
3. Visual model (screenshot analysis)

Constructive interference = sources agree = high confidence
Destructive interference = sources disagree = low confidence, needs more evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

    from src.screen.ocr import OCREngine, OCROutput
    from src.screen.parser import ScreenState, UIElement


# ---------------------------------------------------------------------------
# Dempster-Shafer core
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fused element representation
# ---------------------------------------------------------------------------


@dataclass
class FusedElement:
    """An element enriched by multi-source fusion."""

    index: int
    text: str
    ocr_text: str = ""
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    clickable: bool = False
    scrollable: bool = False
    editable: bool = False
    source: str = "hierarchy"  # hierarchy | ocr | visual | fused
    confidence: float = 1.0

    @property
    def best_text(self) -> str:
        return self.text or self.ocr_text

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )


@dataclass
class FusedScreenState:
    """Screen state after multi-source fusion."""

    activity: str = ""
    package: str = ""
    elements: list[FusedElement] = field(default_factory=list)
    conflict_degree: float = 0.0
    sources_used: list[str] = field(default_factory=list)
    has_scrollable_content: bool = False

    def to_prompt_str(self) -> str:
        lines = [f"Screen: {self.activity} (sources: {', '.join(self.sources_used)})"]
        interactive = [e for e in self.elements if e.clickable or e.scrollable or e.editable]
        lines.append(f"Interactive elements ({len(interactive)}):")
        for e in interactive:
            parts = [f"[{e.index}]"]
            if e.best_text:
                parts.append(f'"{e.best_text}"')
            attrs = []
            if e.clickable:
                attrs.append("clickable")
            if e.scrollable:
                attrs.append("scrollable")
            if e.editable:
                attrs.append("editable")
            if attrs:
                parts.append(f"[{','.join(attrs)}]")
            if e.confidence < 0.9:
                parts.append(f"(conf={e.confidence:.2f})")
            lines.append(f"  {' '.join(parts)}")
        if self.has_scrollable_content:
            lines.append("  [page is scrollable]")
        if self.conflict_degree > 0.3:
            lines.append(f"  [!] High conflict ({self.conflict_degree:.2f}) — sources disagree")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Three-source fusion pipeline
# ---------------------------------------------------------------------------


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Compute IoU between two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fuse_screen(
    hierarchy: ScreenState | None = None,
    ocr: OCROutput | None = None,
    visual_elements: list[dict] | None = None,
) -> FusedScreenState:
    """Fuse up to three screen understanding sources into a unified state.

    Args:
        hierarchy: Parsed UI hierarchy from uiautomator XML.
        ocr: OCR results from screenshot.
        visual_elements: Visual model output (list of dicts with
            keys: text, bounds, type). Optional P2 feature.

    Returns:
        FusedScreenState with all elements and conflict metrics.
    """
    fused_elements: list[FusedElement] = []
    sources_used: list[str] = []
    conflict = 0.0
    idx = 0

    # Step 1: Build base from hierarchy (highest structural reliability)
    ocr_matched: set[int] = set()  # track which OCR results got matched

    if hierarchy and hierarchy.elements:
        sources_used.append("hierarchy")
        for elem in hierarchy.elements:
            fe = FusedElement(
                index=idx,
                text=elem.display_text,
                bounds=elem.bounds,
                clickable=elem.clickable,
                scrollable=elem.scrollable,
                editable=elem.editable,
                source="hierarchy",
                confidence=0.95,
            )
            # Step 2: Enrich with OCR where text is missing
            if ocr and not elem.text and not elem.content_desc:
                for oi, ocr_r in enumerate(ocr.results):
                    if oi in ocr_matched:
                        continue
                    if _bbox_overlap(elem.bounds, ocr_r.bbox) > 0.3:
                        fe.ocr_text = ocr_r.text
                        fe.source = "fused"
                        # Fuse confidence via DS
                        h_mass = EvidenceMass({"element": 0.9, "unknown": 0.1})
                        o_mass = EvidenceMass(
                            {"element": ocr_r.confidence, "unknown": 1 - ocr_r.confidence}
                        )
                        combined, c = dempster_combine(h_mass, o_mass)
                        fe.confidence = combined.beliefs.get("element", 0.5)
                        conflict = max(conflict, c)
                        ocr_matched.add(oi)
                        break
            fused_elements.append(fe)
            idx += 1

    # Step 3: Add OCR-only elements (text not found in hierarchy)
    if ocr:
        if "ocr" not in sources_used:
            sources_used.append("ocr")
        for oi, ocr_r in enumerate(ocr.results):
            if oi in ocr_matched:
                continue
            # OCR-only element — likely a canvas or image-rendered text
            fused_elements.append(
                FusedElement(
                    index=idx,
                    text="",
                    ocr_text=ocr_r.text,
                    bounds=ocr_r.bbox,
                    clickable=True,  # assume tappable
                    source="ocr",
                    confidence=ocr_r.confidence * 0.8,  # lower conf for OCR-only
                )
            )
            idx += 1

    # Step 4: Merge visual model elements (P2 — optional)
    if visual_elements:
        sources_used.append("visual")
        for ve in visual_elements:
            bounds = tuple(ve.get("bounds", (0, 0, 0, 0)))
            # Check if already covered by hierarchy/OCR
            already_covered = any(
                _bbox_overlap(fe.bounds, bounds) > 0.5 for fe in fused_elements
            )
            if not already_covered:
                fused_elements.append(
                    FusedElement(
                        index=idx,
                        text=ve.get("text", ""),
                        bounds=bounds,
                        clickable=ve.get("type", "") in ("button", "link", "icon"),
                        source="visual",
                        confidence=ve.get("confidence", 0.6),
                    )
                )
                idx += 1

    activity = hierarchy.activity if hierarchy else ""
    package = hierarchy.package if hierarchy else ""
    has_scroll = hierarchy.has_scrollable_content if hierarchy else False

    logger.debug(
        f"Fused {len(fused_elements)} elements from {sources_used}, conflict={conflict:.3f}"
    )
    return FusedScreenState(
        activity=activity,
        package=package,
        elements=fused_elements,
        conflict_degree=conflict,
        sources_used=sources_used,
        has_scrollable_content=has_scroll,
    )
