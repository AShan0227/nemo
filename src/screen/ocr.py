"""OCR extraction helpers for non-accessibility text on Android screens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class OCRTextBlock:
    """Single OCR text line with confidence and rough bounds."""

    text: str
    confidence: float
    bounds: tuple[int, int, int, int]  # left, top, right, bottom

    @property
    def normalized_text(self) -> str:
        return _normalize_text(self.text)


def _polygon_to_bounds(points: Any) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    if not isinstance(points, list):
        return (0, 0, 0, 0)

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def _looks_like_entry(item: Any) -> bool:
    if not isinstance(item, list) or len(item) < 2:
        return False
    points = item[0]
    if not isinstance(points, list) or not points:
        return False
    first_point = points[0]
    if not isinstance(first_point, (list, tuple)) or len(first_point) < 2:
        return False
    text_payload = item[1]
    if not isinstance(text_payload, (list, tuple)) or len(text_payload) < 2:
        return False
    if not isinstance(text_payload[0], str):
        return False
    return isinstance(text_payload[1], (int, float))


def parse_paddle_ocr_output(raw_output: Any) -> list[OCRTextBlock]:
    """Parse PaddleOCR output across versions into normalized blocks."""
    blocks: list[OCRTextBlock] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            if _looks_like_entry(node):
                points = node[0]
                text_payload = node[1]
                text = _normalize_text(text_payload[0])
                if not text:
                    return
                confidence = float(text_payload[1])
                bounds = _polygon_to_bounds(points)
                blocks.append(
                    OCRTextBlock(
                        text=text,
                        confidence=_clamp(confidence, 0.0, 1.0),
                        bounds=bounds,
                    )
                )
                return
            for child in node:
                _walk(child)

    _walk(raw_output)
    return blocks


def blocks_to_label_scores(
    blocks: list[OCRTextBlock],
    *,
    min_confidence: float = 0.3,
) -> dict[str, float]:
    """Aggregate OCR blocks into normalized text->confidence map."""
    scores: dict[str, float] = {}
    for block in blocks:
        if block.confidence < min_confidence:
            continue
        key = block.normalized_text.lower()
        if not key:
            continue
        scores[key] = max(scores.get(key, 0.0), block.confidence)
    return scores


class OCRExtractor:
    """Lazy OCR adapter that works when PaddleOCR is available."""

    def __init__(
        self,
        *,
        language: str = "ch",
        use_angle_cls: bool = True,
        min_confidence: float = 0.3,
        enabled: bool = True,
    ) -> None:
        self.language = language
        self.use_angle_cls = use_angle_cls
        self.min_confidence = min_confidence
        self.enabled = enabled
        self._engine: Any | None = None
        self._engine_initialized = False

    @property
    def available(self) -> bool:
        return self._engine is not None

    def extract(self, image: str | Path | Image) -> list[OCRTextBlock]:
        """Extract OCR blocks from screenshot path or PIL image."""
        if not self.enabled:
            return []

        self._ensure_engine()
        if self._engine is None:
            return []

        payload: str | Any
        if isinstance(image, (str, Path)):
            payload = str(image)
        else:
            import numpy as np

            payload = np.asarray(image.convert("RGB"))

        try:
            raw_output = self._engine.ocr(payload, cls=self.use_angle_cls)
        except Exception as exc:
            logger.warning(f"OCR inference failed: {exc}")
            return []

        blocks = parse_paddle_ocr_output(raw_output)
        return [block for block in blocks if block.confidence >= self.min_confidence]

    def _ensure_engine(self) -> None:
        if self._engine_initialized:
            return

        self._engine_initialized = True
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-untyped]
        except ModuleNotFoundError:
            logger.info("PaddleOCR not installed. OCR fallback disabled.")
            self._engine = None
            return

        try:
            self._engine = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.language,
                show_log=False,
            )
        except Exception as exc:
            logger.warning(f"Failed to initialize PaddleOCR engine: {exc}")
            self._engine = None
