"""Screen understanding modules (parser, OCR, and evidence fusion)."""

from src.screen.fusion import (
    EvidenceMass,
    FusionResult,
    SignalCandidate,
    candidates_from_ocr,
    candidates_from_ui,
    candidates_from_visual,
    candidates_to_mass,
    dempster_combine,
    fuse_multiple,
    fuse_screen_sources,
)
from src.screen.ocr import (
    OCRExtractor,
    OCRTextBlock,
    blocks_to_label_scores,
    parse_paddle_ocr_output,
)
from src.screen.parser import (
    ScreenParserCache,
    ScreenState,
    UIElement,
    compute_screen_hash,
    parse_ui_hierarchy,
)

__all__ = [
    "EvidenceMass",
    "FusionResult",
    "OCRExtractor",
    "OCRTextBlock",
    "ScreenParserCache",
    "ScreenState",
    "SignalCandidate",
    "UIElement",
    "blocks_to_label_scores",
    "candidates_from_ocr",
    "candidates_from_ui",
    "candidates_from_visual",
    "candidates_to_mass",
    "compute_screen_hash",
    "dempster_combine",
    "fuse_multiple",
    "fuse_screen_sources",
    "parse_paddle_ocr_output",
    "parse_ui_hierarchy",
]
