"""Tests for visual understanding module."""

import json

from src.screen.visual import (
    GroundingResult,
    VisualAnalysis,
    VisualElement,
    VisualEngine,
    encode_image_base64,
    extract_json,
)


# ---- VisualElement ----

def test_visual_element_center():
    e = VisualElement(text="OK", element_type="button", bounds=(100, 200, 300, 400))
    assert e.center == (200, 300)


def test_analysis_to_label_scores():
    a = VisualAnalysis(elements=[
        VisualElement(text="Send", element_type="button", confidence=0.9),
        VisualElement(text="Cancel", element_type="button", confidence=0.8),
        VisualElement(text="", element_type="icon", description="back arrow", confidence=0.7),
    ])
    scores = a.to_label_scores()
    assert scores["send"] == 0.9
    assert scores["cancel"] == 0.8
    assert scores["back arrow"] == 0.7


def test_analysis_find_by_text():
    a = VisualAnalysis(elements=[
        VisualElement(text="Settings", element_type="text", bounds=(0, 0, 100, 50)),
        VisualElement(text="WiFi", element_type="icon", bounds=(0, 50, 100, 100)),
        VisualElement(text="Bluetooth Settings", element_type="text", bounds=(0, 100, 100, 150)),
    ])
    found = a.find_by_text("settings")
    assert len(found) == 2


# ---- JSON extraction ----

def test_extract_json_direct():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_markdown():
    text = 'Result:\n```json\n{"found": true}\n```'
    assert extract_json(text)["found"] is True


def test_extract_json_embedded():
    text = 'The element is {"x": 200} in screenshot.'
    assert extract_json(text)["x"] == 200


def test_extract_json_none():
    assert extract_json("no json") is None


# ---- Image encoding ----

def test_encode_image_base64():
    from PIL import Image
    import base64
    img = Image.new("RGB", (100, 100), "red")
    b64 = encode_image_base64(img)
    assert len(base64.b64decode(b64)) > 0


def test_encode_image_resize():
    from PIL import Image
    img = Image.new("RGB", (4000, 3000), "blue")
    b64 = encode_image_base64(img, max_size=256)
    assert len(b64) > 0


# ---- VisualEngine parse methods ----

def test_parse_analysis():
    raw = json.dumps({
        "scene": "home screen",
        "app": "Launcher",
        "elements": [
            {"text": "Chrome", "type": "icon", "bounds": [100, 200, 200, 300]},
        ],
    })
    result = VisualEngine._parse_analysis(raw)
    assert result.app_name == "Launcher"
    assert len(result.elements) == 1
    assert result.elements[0].bounds == (100, 200, 200, 300)


def test_parse_analysis_empty():
    result = VisualEngine._parse_analysis("not json")
    assert len(result.elements) == 0


def test_parse_grounding_found():
    raw = json.dumps({"found": True, "x": 250, "y": 350, "bounds": [200, 300, 300, 400], "confidence": 0.92})
    result = VisualEngine._parse_grounding(raw)
    assert result.found and result.x == 250 and result.confidence == 0.92


def test_parse_grounding_not_found():
    result = VisualEngine._parse_grounding(json.dumps({"found": False}))
    assert not result.found


def test_parse_grounding_bad_response():
    result = VisualEngine._parse_grounding("invalid")
    assert not result.found
