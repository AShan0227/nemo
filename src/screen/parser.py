"""UI hierarchy parser — extract structured screen state from Android UI dump."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class UIElement:
    """A single UI element extracted from the screen."""

    index: int
    resource_id: str = ""
    class_name: str = ""
    text: str = ""
    content_desc: str = ""
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)  # left, top, right, bottom
    clickable: bool = False
    scrollable: bool = False
    editable: bool = False
    checked: bool | None = None
    enabled: bool = True
    children: list[UIElement] = field(default_factory=list)

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the element."""
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )

    @property
    def area(self) -> int:
        """Area of the element bounding box."""
        w = max(0, self.bounds[2] - self.bounds[0])
        h = max(0, self.bounds[3] - self.bounds[1])
        return w * h

    @property
    def is_visible(self) -> bool:
        """Whether element has a non-degenerate bounding box."""
        return self.area > 0 and self.enabled

    @property
    def display_text(self) -> str:
        """Best available text representation."""
        return self.text or self.content_desc or self.resource_id.split("/")[-1] or self.class_name

    def to_prompt_str(self) -> str:
        """Format for LLM consumption."""
        parts = [f"[{self.index}]"]
        if self.text:
            parts.append(f'"{self.text}"')
        if self.content_desc and self.content_desc != self.text:
            parts.append(f"({self.content_desc})")
        parts.append(self.class_name.split(".")[-1])
        attrs = []
        if self.clickable:
            attrs.append("clickable")
        if self.scrollable:
            attrs.append("scrollable")
        if self.editable:
            attrs.append("editable")
        if attrs:
            parts.append(f"[{','.join(attrs)}]")
        return " ".join(parts)


@dataclass
class ScreenState:
    """Parsed screen state with all interactive elements."""

    activity: str = ""
    package: str = ""
    elements: list[UIElement] = field(default_factory=list)
    raw_xml: str = ""
    screen_hash: str = ""

    @property
    def interactive_elements(self) -> list[UIElement]:
        """Only elements the user/agent can interact with."""
        return [e for e in self.elements if e.clickable or e.scrollable or e.editable]

    @property
    def scrollable_elements(self) -> list[UIElement]:
        """Elements that support scrolling (may contain off-screen content)."""
        return [e for e in self.elements if e.scrollable]

    @property
    def has_scrollable_content(self) -> bool:
        """Whether the screen likely has content beyond the visible viewport."""
        return len(self.scrollable_elements) > 0

    def to_prompt_str(self) -> str:
        """Simplified screen representation for LLM context."""
        lines = [f"Screen: {self.activity}"]
        lines.append(f"Interactive elements ({len(self.interactive_elements)}):")
        for elem in self.interactive_elements:
            lines.append(f"  {elem.to_prompt_str()}")
        if self.has_scrollable_content:
            lines.append("  [⇕ Page is scrollable — more content may exist below]")
        return "\n".join(lines)


def _parse_bounds(bounds_str: str) -> tuple[int, int, int, int]:
    """Parse '[left,top][right,bottom]' format with fault tolerance."""
    if not bounds_str:
        return (0, 0, 0, 0)
    try:
        # Primary: standard format [l,t][r,b]
        parts = bounds_str.replace("][", ",").strip("[]").split(",")
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except (ValueError, IndexError):
        pass
    # Fallback: regex extraction for malformed strings
    nums = re.findall(r"-?\d+", bounds_str)
    if len(nums) >= 4:
        return (int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3]))
    return (0, 0, 0, 0)


def _sanitize_xml(xml_str: str) -> str:
    """Fix common malformed XML issues from various Android versions."""
    # Remove null bytes
    xml_str = xml_str.replace("\x00", "")
    # Fix unescaped ampersands (common in text attributes)
    xml_str = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#)", "&amp;", xml_str)
    # Remove invalid XML characters (control chars except tab, newline, CR)
    xml_str = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", xml_str)
    return xml_str


def _compute_screen_hash(elements: list[UIElement], activity: str) -> str:
    """Compute a hash representing this screen state for deduplication."""
    sig_parts = [activity]
    for e in elements:
        sig_parts.append(f"{e.resource_id}|{e.text}|{e.bounds}")
    signature = "\n".join(sig_parts)
    return hashlib.md5(signature.encode()).hexdigest()[:12]


def parse_ui_hierarchy(xml_str: str) -> ScreenState:
    """Parse uiautomator XML dump into ScreenState.

    Handles various Android version quirks:
    - BOM / prefix text before XML
    - Malformed attributes (unescaped &, null bytes)
    - Missing or extra attributes
    """
    if not xml_str or not xml_str.strip():
        return ScreenState(raw_xml=xml_str or "")

    # Strip the prefix that uiautomator dump adds
    xml_start = xml_str.find("<?xml")
    if xml_start == -1:
        xml_start = xml_str.find("<hierarchy")
    if xml_start == -1:
        logger.warning("No XML/hierarchy tag found in UI dump")
        return ScreenState(raw_xml=xml_str)

    xml_str = xml_str[xml_start:]
    xml_str = _sanitize_xml(xml_str)

    try:
        root = ET.fromstring(xml_str)
    except ET.ParseError as e:
        logger.error(f"XML parse failed: {e}. Attempting recovery...")
        # Last resort: strip everything after the last closing tag
        last_close = xml_str.rfind("</hierarchy>")
        if last_close != -1:
            xml_str = xml_str[: last_close + len("</hierarchy>")]
            try:
                root = ET.fromstring(xml_str)
            except ET.ParseError:
                logger.error("XML recovery failed, returning empty state")
                return ScreenState(raw_xml=xml_str)
        else:
            return ScreenState(raw_xml=xml_str)

    elements: list[UIElement] = []
    idx = 0

    def _walk(node: ET.Element) -> UIElement | None:
        nonlocal idx
        class_name = node.get("class", "")
        elem = UIElement(
            index=idx,
            resource_id=node.get("resource-id", ""),
            class_name=class_name,
            text=node.get("text", ""),
            content_desc=node.get("content-desc", ""),
            bounds=_parse_bounds(node.get("bounds", "")),
            clickable=node.get("clickable", "false").lower() == "true",
            scrollable=node.get("scrollable", "false").lower() == "true",
            editable=(
                class_name.endswith("EditText")
                or node.get("editable", "false").lower() == "true"
            ),
            checked=node.get("checked") == "true" if node.get("checked") else None,
            enabled=node.get("enabled", "true").lower() != "false",
        )
        idx += 1
        for child in node:
            child_elem = _walk(child)
            if child_elem:
                elem.children.append(child_elem)
        elements.append(elem)
        return elem

    for child in root:
        _walk(child)

    package = root.get("package", "")
    return ScreenState(
        activity=package,
        package=package,
        elements=elements,
        raw_xml=xml_str,
        screen_hash=_compute_screen_hash(elements, package),
    )
