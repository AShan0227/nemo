"""Tests for screen parser."""

from src.screen.parser import ScreenParserCache, compute_screen_hash, parse_ui_hierarchy

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


def test_parse_with_prefix_noise():
    noisy = "UI hierchary dumped to: /dev/tty\n" + SAMPLE_XML
    state = parse_ui_hierarchy(noisy)
    assert len(state.elements) > 0
    assert state.package in {"", "com.android.settings"}


def test_parse_alt_android_attribute_names():
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0">
  <node resourceId="com.example:id/input"
        className="android.widget.EditText"
        contentDescription="Message input"
        text=""
        clickable="true"
        scrollable="false"
        bounds="[10,20][210,120]" />
</hierarchy>"""
    state = parse_ui_hierarchy(xml)
    elem = state.elements[0]
    assert elem.resource_id == "com.example:id/input"
    assert elem.class_name == "android.widget.EditText"
    assert elem.content_desc == "Message input"
    assert elem.editable


def test_screen_hash_is_stable():
    hash_a = compute_screen_hash(SAMPLE_XML)
    hash_b = compute_screen_hash("\n\n" + SAMPLE_XML + "\n")
    assert hash_a == hash_b


def test_screen_parser_cache_dedup():
    cache = ScreenParserCache()
    state_a, dedup_a = cache.parse(SAMPLE_XML)
    state_b, dedup_b = cache.parse(SAMPLE_XML)
    assert not dedup_a
    assert dedup_b
    assert state_a is state_b
