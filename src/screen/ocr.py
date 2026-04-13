"""OCR module — PaddleOCR integration for text recognition on screenshots.

Used as a secondary evidence source when accessibility tree text is missing
or incomplete (e.g., canvas-rendered UI, game screens, image-based content).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image


@dataclass
class OCRResult:
    """A single recognized text region."""

    text: str
    confidence: float
    bbox: tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


@dataclass
class OCROutput:
    """Full OCR results for one screenshot."""

    results: list[OCRResult] = field(default_factory=list)
    language: str = "ch"

    @property
    def all_text(self) -> str:
        """Concatenated text from all regions, sorted top-to-bottom."""
        sorted_results = sorted(self.results, key=lambda r: (r.bbox[1], r.bbox[0]))
        return " ".join(r.text for r in sorted_results)

    def find_text(self, query: str) -> list[OCRResult]:
        """Find regions containing the given text (case-insensitive)."""
        q = query.lower()
        return [r for r in self.results if q in r.text.lower()]


class OCREngine:
    """PaddleOCR wrapper for screen text recognition."""

    def __init__(self, lang: str = "ch", use_gpu: bool = False) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None

    def _ensure_engine(self):
        """Lazy-init PaddleOCR engine on first use."""
        if self._engine is not None:
            return
        try:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=self._use_gpu,
                show_log=False,
            )
            logger.info(f"PaddleOCR initialized (lang={self._lang}, gpu={self._use_gpu})")
        except ImportError:
            logger.warning(
                "PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle"
            )
            raise

    def _bbox_from_poly(self, polygon: list[list[float]]) -> tuple[int, int, int, int]:
        """Convert PaddleOCR polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] to bbox."""
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    def recognize_sync(self, image: Image.Image) -> OCROutput:
        """Run OCR on a PIL Image (synchronous)."""
        import numpy as np

        self._ensure_engine()
        img_array = np.array(image)
        raw = self._engine.ocr(img_array, cls=True)

        results = []
        if raw and raw[0]:
            for line in raw[0]:
                polygon, (text, confidence) = line
                bbox = self._bbox_from_poly(polygon)
                results.append(OCRResult(text=text, confidence=confidence, bbox=bbox))

        logger.debug(f"OCR found {len(results)} text regions")
        return OCROutput(results=results, language=self._lang)

    async def recognize(self, image: Image.Image) -> OCROutput:
        """Run OCR on a PIL Image (async wrapper)."""
        return await asyncio.to_thread(self.recognize_sync, image)
