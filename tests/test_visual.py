"""Tests for visual understanding module (unit tests without actual LLM calls)."""

import json

from src.screen.visual import (
    GroundingResult,
    VisualAnalysis,
    VisualElement,
    VisualEngine,
    _bbox_iou,
    _encode_image_base64,
    _extract_json,
)


# ---- VisualElement ----

def test_visual_element_center():
    e = VisualElement(text="OK", element_type="button", bounds=(100, 200, 300, 400))
    assert e.center == (200, 300)


def test_visual_element_to_fusion_dict():
    e = VisualElement(text="Submit", element_type="button", bounds=(0, 0, 100, 50), confidence=0.9)
    d = e.to_fusion_dict()
    assert d["text"] == "Submit"
    assert d["type"] == "button"
    assert d["bounds"] == (0, 0, 100, 50)
    assert d["confidence"] == 0.9


# ---- VisualAnalysis ----

def test_analysis_find_by_text():
    a = VisualAnalysis(elements=[
        VisualElement(text="Settings", element_type="text", bounds=(0, 0, 100, 50)),
        VisualElement(text="WiFi", element_type="icon", bounds=(0, 50, 100, 100)),
        VisualElement(text="Bluetooth Settings", element_type="text", bounds=(0, 100, 100, 150)),
    ])
    found = a.find_by_text("settings")
    assert len(found) == 2


def test_analysis_find_by_type():
    a = VisualAnalysis(elements=[
        VisualElement(text="OK", element_type="button", bounds=(0, 0, 100, 50)),
        VisualElement(text="Cancel", element_type="button", bounds=(100, 0, 200, 50)),
        VisualElement(text="Title", element_type="text", bounds=(0, 50, 200, 100)),
    ])
    buttons = a.find_by_type("button")
    assert len(buttons) == 2


def test_analysis_find_by_description():
    a = VisualAnalysis(elements=[
        VisualElement(text="", element_type="icon", bounds=(0, 0, 50, 50),
                      description="hamburger menu icon"),
    ])
    found = a.find_by_text("hamburger")
    assert len(found) == 1


# ---- JSON extraction ----

def test_extract_json_direct():
    data = _extract_json('{"found": true, "x": 100}')
    assert data == {"found": True, "x": 100}


def test_extract_json_from_markdown():
    text = 'Here is the result:\n```json\n{"found": true, "x": 100}\n```\nDone.'
    data = _extract_json(text)
    assert data["found"] is True


def test_extract_json_from_text():
    text = 'The element is at {"found": true, "x": 200, "y": 300} in the screenshot.'
    data = _extract_json(text)
    assert data["x"] == 200


def test_extract_json_none():
    assert _extract_json("no json here at all") is None


def test_extract_json_malformed():
    assert _extract_json("{broken json!!!") is None


# ---- Bounding box IoU ----

def test_bbox_iou_full():
    assert _bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_bbox_iou_none():
    assert _bbox_iou((0, 0, 50, 50), (100, 100, 200, 200)) == 0.0


def test_bbox_iou_partial():
    iou = _bbox_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0.1 < iou < 0.5


def test_bbox_iou_zero_area():
    assert _bbox_iou((0, 0, 0, 0), (0, 0, 100, 100)) == 0.0


# ---- Image encoding ----

def test_encode_image_base64():
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="red")
    b64 = _encode_image_base64(img)
    assert len(b64) > 0
    # Should be valid base64
    import base64
    decoded = base64.b64decode(b64)
    assert len(decoded) > 0


def test_encode_image_resize():
    from PIL import Image
    img = Image.new("RGB", (4000, 3000), color="blue")
    b64 = _encode_image_base64(img, max_size=512)
    # Should still produce valid base64 (just smaller)
    assert len(b64) > 0


# ---- VisualEngine parse methods ----

def test_parse_analysis():
    raw = json.dumps({
        "scene": "home screen",
        "app": "Launcher",
        "elements": [
            {"text": "Chrome", "type": "icon", "bounds": [100, 200, 200, 300],
             "description": "Chrome browser"},
            {"text": "Settings", "type": "icon", "bounds": [300, 200, 400, 300],
             "description": "System settings"},
        ],
    })
    result = VisualEngine._parse_analysis(raw)
    assert result.scene_description == "home screen"
    assert result.app_name == "Launcher"
    assert len(result.elements) == 2
    assert result.elements[0].text == "Chrome"
    assert result.elements[0].bounds == (100, 200, 200, 300)


def test_parse_analysis_empty():
    result = VisualEngine._parse_analysis("not json")
    assert len(result.elements) == 0
    assert result.raw_response == "not json"


def test_parse_analysis_from_markdown():
    raw = """Here's the analysis:
```json
{
  "scene": "WeChat chat list",
  "app": "WeChat",
  "elements": [
    {"text": "张三", "type": "text", "bounds": [0, 100, 500, 150], "description": "contact name"}
  ]
}
```"""
    result = VisualEngine._parse_analysis(raw)
    assert result.app_name == "WeChat"
    assert len(result.elements) == 1


def test_parse_grounding_found():
    raw = json.dumps({
        "found": True,
        "x": 250,
        "y": 350,
        "bounds": [200, 300, 300, 400],
        "confidence": 0.92,
        "description": "search bar at top",
    })
    result = VisualEngine._parse_grounding(raw)
    assert result.found is True
    assert result.x == 250
    assert result.y == 350
    assert result.confidence == 0.92


def test_parse_grounding_not_found():
    raw = json.dumps({"found": False})
    result = VisualEngine._parse_grounding(raw)
    assert result.found is False


def test_parse_grounding_bad_response():
    result = VisualEngine._parse_grounding("I cannot help")
    assert result.found is False


# ---- GroundingResult ----

def test_grounding_result_defaults():
    r = GroundingResult(found=False)
    assert r.x == 0
    assert r.y == 0
    assert r.confidence == 0.0
