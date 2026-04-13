"""Tests for OCR module (unit tests without PaddleOCR dependency)."""

from src.screen.ocr import OCRResult, OCROutput


def test_ocr_result_center():
    r = OCRResult(text="Hello", confidence=0.95, bbox=(100, 200, 300, 400))
    assert r.center == (200, 300)


def test_ocr_output_all_text():
    output = OCROutput(results=[
        OCRResult(text="Line 2", confidence=0.9, bbox=(0, 100, 100, 200)),
        OCRResult(text="Line 1", confidence=0.9, bbox=(0, 0, 100, 50)),
    ])
    # Should be sorted top-to-bottom
    assert output.all_text == "Line 1 Line 2"


def test_ocr_output_find_text():
    output = OCROutput(results=[
        OCRResult(text="Settings", confidence=0.9, bbox=(0, 0, 100, 50)),
        OCRResult(text="WiFi", confidence=0.85, bbox=(0, 50, 100, 100)),
        OCRResult(text="Bluetooth Settings", confidence=0.8, bbox=(0, 100, 100, 150)),
    ])
    found = output.find_text("settings")
    assert len(found) == 2  # "Settings" and "Bluetooth Settings"


def test_ocr_output_find_text_case_insensitive():
    output = OCROutput(results=[
        OCRResult(text="HELLO World", confidence=0.9, bbox=(0, 0, 100, 50)),
    ])
    assert len(output.find_text("hello")) == 1


def test_ocr_output_empty():
    output = OCROutput()
    assert output.all_text == ""
    assert output.find_text("anything") == []
