"""UI hierarchy parser — extract structured screen state from Android UI dump."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field


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

    @property
    def interactive_elements(self) -> list[UIElement]:
        """Only elements the user/agent can interact with."""
        return [e for e in self.elements if e.clickable or e.scrollable or e.editable]

    def to_prompt_str(self) -> str:
        """Simplified screen representation for LLM context."""
        lines = [f"Screen: {self.activity}"]
        lines.append(f"Interactive elements ({len(self.interactive_elements)}):")
        for elem in self.interactive_elements:
            lines.append(f"  {elem.to_prompt_str()}")
        return "\n".join(lines)


def _parse_bounds(bounds_str: str) -> tuple[int, int, int, int]:
    """Parse '[left,top][right,bottom]' format."""
    try:
        parts = bounds_str.replace("][", ",").strip("[]").split(",")
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except (ValueError, IndexError):
        return (0, 0, 0, 0)


def parse_ui_hierarchy(xml_str: str) -> ScreenState:
    """Parse uiautomator XML dump into ScreenState."""
    # Strip the prefix that uiautomator dump adds
    xml_start = xml_str.find("<?xml")
    if xml_start == -1:
        xml_start = xml_str.find("<hierarchy")
    if xml_start == -1:
        return ScreenState(raw_xml=xml_str)

    xml_str = xml_str[xml_start:]
    root = ET.fromstring(xml_str)
    elements: list[UIElement] = []
    idx = 0

    def _walk(node: ET.Element) -> UIElement | None:
        nonlocal idx
        elem = UIElement(
            index=idx,
            resource_id=node.get("resource-id", ""),
            class_name=node.get("class", ""),
            text=node.get("text", ""),
            content_desc=node.get("content-desc", ""),
            bounds=_parse_bounds(node.get("bounds", "")),
            clickable=node.get("clickable") == "true",
            scrollable=node.get("scrollable") == "true",
            editable=node.get("class", "").endswith("EditText"),
            checked=node.get("checked") == "true" if node.get("checked") else None,
            enabled=node.get("enabled", "true") == "true",
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
    )
