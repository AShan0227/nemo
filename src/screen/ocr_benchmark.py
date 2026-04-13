"""OCR accuracy benchmark framework.

Measures PaddleOCR recognition accuracy on phone screenshots with
ground-truth labels. Used to tune det_db_thresh, rec_batch_num,
and min_confidence parameters per app/scenario.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.screen.ocr import OCRExtractor, OCRTextBlock


@dataclass
class OCRGroundTruth:
    """Expected text in a screenshot with approximate location."""

    text: str
    bounds: tuple[int, int, int, int] | None = None  # optional expected bbox


@dataclass
class OCRBenchmarkResult:
    """Result of OCR accuracy evaluation on a single image."""

    image_path: str
    expected: list[str]
    detected: list[str]
    matched: list[str]
    missed: list[str]
    extra: list[str]  # detected but not expected
    accuracy: float  # matched / expected
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "expected": self.expected,
            "detected": self.detected,
            "matched": self.matched,
            "missed": self.missed,
            "extra": self.extra,
            "accuracy": self.accuracy,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class OCRBenchmarkReport:
    """Aggregated benchmark across multiple images."""

    results: list[OCRBenchmarkResult] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.accuracy for r in self.results) / len(self.results)

    @property
    def avg_elapsed_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.elapsed_ms for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "overall_accuracy": self.overall_accuracy,
            "avg_elapsed_ms": self.avg_elapsed_ms,
            "num_images": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        logger.info(f"Benchmark saved: {path}")


def evaluate_ocr_image(
    image_path: str | Path,
    ground_truth: list[str],
    extractor: OCRExtractor,
) -> OCRBenchmarkResult:
    """Evaluate OCR accuracy on a single image.

    Args:
        image_path: Path to screenshot PNG.
        ground_truth: List of expected text strings.
        extractor: OCRExtractor with desired params.

    Returns:
        OCRBenchmarkResult with accuracy metrics.
    """
    from PIL import Image

    image = Image.open(str(image_path))

    start = time.perf_counter()
    blocks = extractor.extract(image)
    elapsed_ms = (time.perf_counter() - start) * 1000

    detected = [b.text.strip() for b in blocks]
    detected_lower = {d.lower() for d in detected}

    matched = []
    missed = []
    for gt in ground_truth:
        if gt.lower() in detected_lower:
            matched.append(gt)
        else:
            # Fuzzy: check if gt is a substring of any detection
            if any(gt.lower() in d for d in detected_lower):
                matched.append(gt)
            else:
                missed.append(gt)

    gt_lower = {g.lower() for g in ground_truth}
    extra = [d for d in detected if d.lower() not in gt_lower]

    accuracy = len(matched) / len(ground_truth) if ground_truth else 1.0

    return OCRBenchmarkResult(
        image_path=str(image_path),
        expected=ground_truth,
        detected=detected,
        matched=matched,
        missed=missed,
        extra=extra,
        accuracy=accuracy,
        elapsed_ms=elapsed_ms,
    )


def run_benchmark(
    test_cases: list[tuple[str | Path, list[str]]],
    *,
    det_db_thresh: float = 0.3,
    rec_batch_num: int = 6,
    min_confidence: float = 0.3,
) -> OCRBenchmarkReport:
    """Run OCR benchmark across multiple test images.

    Args:
        test_cases: List of (image_path, ground_truth_texts).
        det_db_thresh: Detection threshold to test.
        rec_batch_num: Batch size for recognition.
        min_confidence: Minimum confidence to accept.

    Returns:
        OCRBenchmarkReport with per-image and aggregate results.
    """
    extractor = OCRExtractor(
        enabled=True,
        det_db_thresh=det_db_thresh,
        rec_batch_num=rec_batch_num,
        min_confidence=min_confidence,
    )

    report = OCRBenchmarkReport(
        params={
            "det_db_thresh": det_db_thresh,
            "rec_batch_num": rec_batch_num,
            "min_confidence": min_confidence,
        }
    )

    for image_path, ground_truth in test_cases:
        result = evaluate_ocr_image(image_path, ground_truth, extractor)
        report.results.append(result)
        logger.info(
            f"OCR [{Path(image_path).name}]: "
            f"acc={result.accuracy:.2f} ({len(result.matched)}/{len(result.expected)}) "
            f"in {result.elapsed_ms:.0f}ms"
        )

    logger.info(
        f"Benchmark: overall_acc={report.overall_accuracy:.2f}, "
        f"avg={report.avg_elapsed_ms:.0f}ms, "
        f"n={len(report.results)}"
    )
    return report
