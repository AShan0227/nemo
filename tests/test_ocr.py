"""Tests for OCR adapter and output parsing."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.screen.ocr import (
    OCRExtractor,
    OCRTextBlock,
    blocks_to_label_scores,
    parse_paddle_ocr_output,
)


def test_parse_paddle_ocr_output_basic():
    raw = [
        [
            [[0, 0], [20, 0], [20, 10], [0, 10]],
            ("Hello", 0.92),
        ],
        [
            [[0, 20], [30, 20], [30, 30], [0, 30]],
            ("World", 0.80),
        ],
    ]
    blocks = parse_paddle_ocr_output(raw)
    assert len(blocks) == 2
    assert blocks[0].text == "Hello"
    assert blocks[0].bounds == (0, 0, 20, 10)


def test_parse_paddle_ocr_output_handles_nested_batches():
    raw = [
        [
            [
                [[1, 2], [11, 2], [11, 8], [1, 8]],
                (" Nested  Text ", 0.77),
            ]
        ]
    ]
    blocks = parse_paddle_ocr_output(raw)
    assert len(blocks) == 1
    assert blocks[0].text == "Nested Text"


def test_blocks_to_label_scores_aggregates_best_confidence():
    blocks = [
        OCRTextBlock(text="Search", confidence=0.61, bounds=(0, 0, 10, 10)),
        OCRTextBlock(text="search", confidence=0.85, bounds=(0, 0, 10, 10)),
        OCRTextBlock(text="Low", confidence=0.2, bounds=(0, 0, 10, 10)),
    ]
    scores = blocks_to_label_scores(blocks)
    assert scores["search"] == 0.85
    assert "low" not in scores


def test_extractor_disabled_returns_empty():
    extractor = OCRExtractor(enabled=False)
    result = extractor.extract(Image.new("RGB", (20, 20)))
    assert result == []


class FakeOCREngine:
    def ocr(self, payload: object, cls: bool = True):
        assert payload is not None
        assert cls
        return [
            [
                [
                    [[0, 0], [16, 0], [16, 8], [0, 8]],
                    ("High", 0.93),
                ],
                [
                    [[0, 10], [12, 10], [12, 16], [0, 16]],
                    ("Low", 0.10),
                ],
            ]
        ]


def test_extractor_with_fake_engine_filters_by_confidence(tmp_path: Path):
    extractor = OCRExtractor(enabled=True, min_confidence=0.5)
    extractor._engine = FakeOCREngine()  # type: ignore[attr-defined]
    extractor._engine_initialized = True  # type: ignore[attr-defined]

    image = Image.new("RGB", (20, 20), (255, 255, 255))
    blocks = extractor.extract(image)

    assert len(blocks) == 1
    assert blocks[0].text == "High"
