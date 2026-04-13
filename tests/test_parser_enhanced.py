"""Tests for enhanced screen parser — WebView, modal, occlusion detection."""

from src.screen.parser import parse_ui_hierarchy


WEBVIEW_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.webkit.WebView" bounds="[0,0][1080,1920]"
        clickable="false" scrollable="true" enabled="true" text="">
    <node index="0" class="android.view.View" text="Web Content"
          clickable="true" bounds="[10,10][500,60]" enabled="true" />
  </node>
</hierarchy>"""

MODAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.FrameLayout" bounds="[0,0][1080,1920]"
        clickable="false" enabled="true">
    <node index="0" class="android.widget.Button" text="Background"
          clickable="true" bounds="[10,10][200,60]" enabled="true" />
    <node index="1" class="android.app.AlertDialog" bounds="[100,400][980,1000]"
          clickable="false" enabled="true">
      <node index="0" class="android.widget.Button" text="OK"
            clickable="true" bounds="[400,900][600,960]" enabled="true" />
      <node index="1" class="android.widget.Button" text="Cancel"
            clickable="true" bounds="[200,900][400,960]" enabled="true" />
    </node>
  </node>
</hierarchy>"""

OCCLUSION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.FrameLayout" bounds="[0,0][1080,1920]"
        clickable="false" enabled="true">
    <node index="0" class="android.widget.Button" text="Hidden Button"
          clickable="true" bounds="[100,100][500,200]" enabled="true" />
    <node index="1" class="android.widget.FrameLayout" bounds="[0,0][1080,1920]"
          clickable="false" enabled="true">
      <node index="0" class="android.widget.TextView" text="Overlay"
            clickable="false" bounds="[0,0][1080,1920]" enabled="true" />
    </node>
  </node>
</hierarchy>"""


def test_webview_detection():
    state = parse_ui_hierarchy(WEBVIEW_XML)
    assert state.has_webview
    assert len(state.webview_elements) == 1


def test_no_webview():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.FrameLayout" bounds="[0,0][1080,1920]"
        clickable="false" enabled="true" />
</hierarchy>"""
    state = parse_ui_hierarchy(xml)
    assert not state.has_webview


def test_webview_prompt_hint():
    state = parse_ui_hierarchy(WEBVIEW_XML)
    prompt = state.to_prompt_str()
    assert "WebView" in prompt


def test_modal_detection():
    state = parse_ui_hierarchy(MODAL_XML)
    modals = state.detect_modals()
    assert len(modals) == 1
    assert "AlertDialog" in modals[0].class_name


def test_has_modal():
    state = parse_ui_hierarchy(MODAL_XML)
    assert state.has_modal


def test_modal_prompt_hint():
    state = parse_ui_hierarchy(MODAL_XML)
    prompt = state.to_prompt_str()
    assert "Modal" in prompt


def test_no_modal():
    state = parse_ui_hierarchy(WEBVIEW_XML)
    assert not state.has_modal


def test_occlusion_detection():
    state = parse_ui_hierarchy(OCCLUSION_XML)
    occluded = state.detect_occluded_elements()
    # "Hidden Button" should be detected as occluded by the full-screen overlay
    assert len(occluded) >= 1
    hidden = [pair for pair in occluded if pair[0].text == "Hidden Button"]
    assert len(hidden) >= 1


def test_occlusion_prompt_hint():
    state = parse_ui_hierarchy(OCCLUSION_XML)
    prompt = state.to_prompt_str()
    assert "occluded" in prompt.lower()


def test_no_occlusion():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" class="android.widget.Button" text="A"
        clickable="true" bounds="[0,0][100,50]" enabled="true" />
  <node index="1" class="android.widget.Button" text="B"
        clickable="true" bounds="[200,0][300,50]" enabled="true" />
</hierarchy>"""
    state = parse_ui_hierarchy(xml)
    assert len(state.detect_occluded_elements()) == 0
