"""Tests for screen parser."""

from src.screen.parser import parse_ui_hierarchy


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node index="0" text="" resource-id="" class="android.widget.FrameLayout"
        content-desc="" checkable="false" checked="false" clickable="false"
        enabled="true" focusable="false" bounds="[0,0][1080,1920]">
    <node index="0" text="Settings" resource-id="com.android.settings:id/title"
          class="android.widget.TextView" content-desc=""
          clickable="true" enabled="true" bounds="[0,0][540,100]" />
    <node index="1" text="" resource-id="com.android.settings:id/search"
          class="android.widget.EditText" content-desc="Search settings"
          clickable="true" enabled="true" bounds="[0,100][1080,200]" />
  </node>
</hierarchy>"""


def test_parse_basic():
    state = parse_ui_hierarchy(SAMPLE_XML)
    assert len(state.elements) > 0


def test_interactive_elements():
    state = parse_ui_hierarchy(SAMPLE_XML)
    interactive = state.interactive_elements
    assert len(interactive) >= 2  # Settings text + search


def test_element_center():
    state = parse_ui_hierarchy(SAMPLE_XML)
    for elem in state.elements:
        if elem.text == "Settings":
            assert elem.center == (270, 50)
            break


def test_prompt_output():
    state = parse_ui_hierarchy(SAMPLE_XML)
    prompt = state.to_prompt_str()
    assert "Settings" in prompt
    assert "Interactive elements" in prompt
