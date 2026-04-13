"""Tests for screen parser."""

from src.screen.parser import UIElement, parse_ui_hierarchy, _parse_bounds, _sanitize_xml


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

SCROLLABLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<hierarchy rotation="0" package="com.test">
  <node index="0" text="" class="android.widget.ScrollView"
        scrollable="true" bounds="[0,0][1080,1920]" clickable="false" enabled="true">
    <node index="0" text="Item 1" class="android.widget.TextView"
          clickable="true" enabled="true" bounds="[0,0][1080,100]" />
    <node index="1" text="Item 2" class="android.widget.TextView"
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


def test_screen_hash_generated():
    state = parse_ui_hierarchy(SAMPLE_XML)
    assert state.screen_hash
    assert len(state.screen_hash) == 12


def test_screen_hash_deterministic():
    state1 = parse_ui_hierarchy(SAMPLE_XML)
    state2 = parse_ui_hierarchy(SAMPLE_XML)
    assert state1.screen_hash == state2.screen_hash


def test_screen_hash_differs():
    state1 = parse_ui_hierarchy(SAMPLE_XML)
    state2 = parse_ui_hierarchy(SCROLLABLE_XML)
    assert state1.screen_hash != state2.screen_hash


def test_empty_input():
    state = parse_ui_hierarchy("")
    assert len(state.elements) == 0


def test_garbage_input():
    state = parse_ui_hierarchy("not xml at all %%% garbage")
    assert len(state.elements) == 0


def test_prefix_stripping():
    prefixed = "UI hierchary dumped to: /dev/tty\n" + SAMPLE_XML
    state = parse_ui_hierarchy(prefixed)
    assert len(state.elements) > 0


def test_parse_bounds_standard():
    assert _parse_bounds("[0,0][1080,1920]") == (0, 0, 1080, 1920)


def test_parse_bounds_empty():
    assert _parse_bounds("") == (0, 0, 0, 0)


def test_parse_bounds_malformed():
    assert _parse_bounds("[10,20,30,40]") == (10, 20, 30, 40)


def test_parse_bounds_negative():
    assert _parse_bounds("[-5,10][100,200]") == (-5, 10, 100, 200)


def test_sanitize_xml_ampersand():
    result = _sanitize_xml('<node text="A & B" />')
    assert "&amp;" in result


def test_sanitize_xml_null_bytes():
    result = _sanitize_xml("hello\x00world")
    assert "\x00" not in result


def test_sanitize_xml_control_chars():
    result = _sanitize_xml("hello\x03world")
    assert "\x03" not in result


def test_element_area():
    elem = UIElement(index=0, bounds=(0, 0, 100, 200))
    assert elem.area == 20000


def test_element_area_degenerate():
    elem = UIElement(index=0, bounds=(100, 100, 100, 100))
    assert elem.area == 0


def test_element_is_visible():
    elem = UIElement(index=0, bounds=(0, 0, 100, 200), enabled=True)
    assert elem.is_visible

    disabled = UIElement(index=0, bounds=(0, 0, 100, 200), enabled=False)
    assert not disabled.is_visible

    zero_area = UIElement(index=0, bounds=(0, 0, 0, 0), enabled=True)
    assert not zero_area.is_visible


def test_scrollable_detection():
    state = parse_ui_hierarchy(SCROLLABLE_XML)
    assert state.has_scrollable_content
    assert len(state.scrollable_elements) >= 1


def test_scrollable_prompt_hint():
    state = parse_ui_hierarchy(SCROLLABLE_XML)
    prompt = state.to_prompt_str()
    assert "scrollable" in prompt.lower()


def test_editable_detection():
    state = parse_ui_hierarchy(SAMPLE_XML)
    editable = [e for e in state.elements if e.editable]
    assert len(editable) >= 1  # EditText should be editable


def test_xml_recovery_truncated():
    """Test recovery when XML is truncated after hierarchy close."""
    truncated = SAMPLE_XML + "\nsome garbage after"
    state = parse_ui_hierarchy(truncated)
    assert len(state.elements) > 0
