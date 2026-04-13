"""Tests for Dempster-Shafer fusion and three-source fusion pipeline."""

from src.screen.fusion import (
    EvidenceMass,
    FusedElement,
    FusedScreenState,
    dempster_combine,
    fuse_multiple,
    fuse_screen,
    _bbox_overlap,
)
from src.screen.parser import ScreenState, UIElement, parse_ui_hierarchy
from src.screen.ocr import OCRResult, OCROutput


# ---- Dempster-Shafer core ----

def test_agreeing_sources():
    m1 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    m2 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    combined, conflict = dempster_combine(m1, m2)
    assert combined.beliefs["button"] > 0.9
    assert conflict < 0.1


def test_conflicting_sources():
    m1 = EvidenceMass({"button": 0.9, "unknown": 0.1})
    m2 = EvidenceMass({"text": 0.9, "unknown": 0.1})
    combined, conflict = dempster_combine(m1, m2)
    assert conflict > 0.5


def test_multiple_fusion():
    m1 = EvidenceMass({"button": 0.7, "unknown": 0.3})
    m2 = EvidenceMass({"button": 0.6, "unknown": 0.4})
    m3 = EvidenceMass({"button": 0.8, "unknown": 0.2})
    combined, _ = fuse_multiple(m1, m2, m3)
    assert combined.beliefs["button"] > 0.8


def test_empty_fusion():
    combined, conflict = fuse_multiple()
    assert combined.beliefs.get("unknown") == 1.0
    assert conflict == 0.0


# ---- Bounding box overlap ----

def test_bbox_overlap_full():
    assert _bbox_overlap((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_bbox_overlap_none():
    assert _bbox_overlap((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0


def test_bbox_overlap_partial():
    iou = _bbox_overlap((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.1 < iou < 0.5


def test_bbox_overlap_zero_area():
    assert _bbox_overlap((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


# ---- Three-source fusion pipeline ----

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0" package="com.test">
  <node index="0" text="Hello" class="android.widget.TextView"
        clickable="true" enabled="true" bounds="[0,0][200,50]" />
  <node index="1" text="" class="android.widget.ImageView"
        clickable="true" enabled="true" bounds="[0,50][200,100]"
        content-desc="" />
</hierarchy>"""


def test_fuse_hierarchy_only():
    hierarchy = parse_ui_hierarchy(SAMPLE_XML)
    fused = fuse_screen(hierarchy=hierarchy)
    assert len(fused.elements) > 0
    assert "hierarchy" in fused.sources_used
    assert fused.conflict_degree == 0.0


def test_fuse_hierarchy_with_ocr():
    hierarchy = parse_ui_hierarchy(SAMPLE_XML)
    # OCR finds text in the image area (element index 1 has no text)
    ocr = OCROutput(results=[
        OCRResult(text="Click Here", confidence=0.95, bbox=(10, 55, 190, 95)),
    ])
    fused = fuse_screen(hierarchy=hierarchy, ocr=ocr)
    assert "hierarchy" in fused.sources_used
    # OCR text should be matched to the image element
    enriched = [e for e in fused.elements if e.ocr_text == "Click Here"]
    assert len(enriched) == 1


def test_fuse_ocr_only_elements():
    hierarchy = parse_ui_hierarchy(SAMPLE_XML)
    # OCR finds text in an area not covered by hierarchy
    ocr = OCROutput(results=[
        OCRResult(text="Canvas Text", confidence=0.85, bbox=(300, 300, 500, 350)),
    ])
    fused = fuse_screen(hierarchy=hierarchy, ocr=ocr)
    # Should have an OCR-only element
    ocr_only = [e for e in fused.elements if e.source == "ocr"]
    assert len(ocr_only) == 1
    assert ocr_only[0].ocr_text == "Canvas Text"


def test_fuse_with_visual_elements():
    hierarchy = parse_ui_hierarchy(SAMPLE_XML)
    visual = [
        {"text": "Hidden Button", "bounds": (500, 500, 700, 550), "type": "button", "confidence": 0.7},
    ]
    fused = fuse_screen(hierarchy=hierarchy, visual_elements=visual)
    assert "visual" in fused.sources_used
    visual_elems = [e for e in fused.elements if e.source == "visual"]
    assert len(visual_elems) == 1


def test_fuse_visual_dedup():
    """Visual element overlapping hierarchy should be skipped."""
    hierarchy = parse_ui_hierarchy(SAMPLE_XML)
    visual = [
        {"text": "Hello Dup", "bounds": (0, 0, 200, 50), "type": "text", "confidence": 0.9},
    ]
    fused = fuse_screen(hierarchy=hierarchy, visual_elements=visual)
    visual_elems = [e for e in fused.elements if e.source == "visual"]
    assert len(visual_elems) == 0


def test_fuse_empty():
    fused = fuse_screen()
    assert len(fused.elements) == 0
    assert len(fused.sources_used) == 0


# ---- FusedScreenState ----

def test_fused_prompt_str():
    fused = FusedScreenState(
        activity="com.test",
        elements=[
            FusedElement(index=0, text="Button", clickable=True, confidence=0.95),
            FusedElement(index=1, text="Low Conf", clickable=True, confidence=0.5),
        ],
        sources_used=["hierarchy", "ocr"],
    )
    prompt = fused.to_prompt_str()
    assert "hierarchy" in prompt
    assert "conf=0.50" in prompt  # low confidence shown


def test_fused_prompt_conflict_warning():
    fused = FusedScreenState(
        activity="com.test",
        conflict_degree=0.6,
        sources_used=["hierarchy"],
    )
    prompt = fused.to_prompt_str()
    assert "conflict" in prompt.lower()
