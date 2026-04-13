"""Tests for Dempster-Shafer fusion."""

from src.screen.fusion import (
    EvidenceMass,
    candidates_from_ocr,
    candidates_from_ui,
    candidates_from_visual,
    dempster_combine,
    fuse_multiple,
    fuse_screen_sources,
)
from src.screen.ocr import OCRTextBlock
from src.screen.parser import UIElement


def test_agreeing_sources():
    m1 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    m2 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    combined, conflict = dempster_combine(m1, m2)
    assert combined.beliefs["button"] > 0.9
    assert conflict < 0.1


def test_conflicting_sources():
    m1 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    m2 = EvidenceMass({"text": 0.9, "unknown": 0.1})
    combined, conflict = fuse_multiple(m1, m2)
    assert conflict > 0.5


def test_multiple_fusion():
    m1 = EvidenceMass({"button": 0.7, "unknown": 0.3})
    m2 = EvidenceMass({"button": 0.6, "unknown": 0.4})
    m3 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    combined, _ = fuse_multiple(m1, m2, m3)
    assert combined.beliefs["button"] > 0.8


def test_fuse_screen_sources_prefers_cross_source_agreement():
    ui_elements = [
        UIElement(
            index=0,
            text="Send",
            class_name="android.widget.Button",
            clickable=True,
            bounds=(0, 0, 50, 20),
        ),
        UIElement(
            index=1,
            text="Cancel",
            class_name="android.widget.Button",
            clickable=True,
            bounds=(50, 0, 100, 20),
        ),
    ]
    ocr_blocks = [OCRTextBlock(text="send", confidence=0.94, bounds=(1, 1, 49, 19))]
    visual = {"send": 0.78, "dialog": 0.65}

    result = fuse_screen_sources(
        ui_candidates=candidates_from_ui(ui_elements),
        ocr_candidates=candidates_from_ocr(ocr_blocks),
        visual_candidates=candidates_from_visual(visual),
    )
    assert result.best_label == "send"
    assert result.conflict < 0.5
    assert "ui" in result.source_masses
    assert "ocr" in result.source_masses
    assert "visual" in result.source_masses


def test_fuse_screen_sources_handles_empty_inputs():
    result = fuse_screen_sources()
    assert result.best_label == "unknown"
    assert result.conflict == 0.0
