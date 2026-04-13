"""Tests for OCR benchmark framework."""

from src.screen.ocr_benchmark import (
    OCRBenchmarkReport,
    OCRBenchmarkResult,
    evaluate_ocr_image,
)
from src.screen.ocr import OCRExtractor


def test_benchmark_result_accuracy():
    r = OCRBenchmarkResult(
        image_path="test.png",
        expected=["Hello", "World", "Missing"],
        detected=["Hello", "World", "Extra"],
        matched=["Hello", "World"],
        missed=["Missing"],
        extra=["Extra"],
        accuracy=2 / 3,
        elapsed_ms=50.0,
    )
    assert abs(r.accuracy - 0.6667) < 0.01
    d = r.to_dict()
    assert d["missed"] == ["Missing"]


def test_benchmark_report_overall():
    report = OCRBenchmarkReport(
        results=[
            OCRBenchmarkResult("a.png", ["A"], ["A"], ["A"], [], [], 1.0, 10),
            OCRBenchmarkResult("b.png", ["B", "C"], ["B"], ["B"], ["C"], [], 0.5, 20),
        ],
        params={"det_db_thresh": 0.3},
    )
    assert abs(report.overall_accuracy - 0.75) < 0.01
    assert report.avg_elapsed_ms == 15.0


def test_benchmark_report_empty():
    report = OCRBenchmarkReport()
    assert report.overall_accuracy == 0.0
    assert report.avg_elapsed_ms == 0.0


def test_benchmark_report_save(tmp_path):
    report = OCRBenchmarkReport(
        results=[OCRBenchmarkResult("x.png", ["T"], ["T"], ["T"], [], [], 1.0, 5)],
        params={"min_confidence": 0.5},
    )
    path = tmp_path / "report.json"
    report.save(path)
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["overall_accuracy"] == 1.0


def test_benchmark_result_to_dict_fields():
    r = OCRBenchmarkResult("i.png", [], [], [], [], [], 1.0, 0.0)
    d = r.to_dict()
    assert "image_path" in d
    assert "accuracy" in d
    assert "elapsed_ms" in d
