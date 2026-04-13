"""UI hierarchy parser — extract structured screen state from Android UI dump."""

from __future__ import annotations

import hashlib
import re
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
    fusion_confidence: float = 1.0  # from DS fusion (1.0 = no conflict)
    fusion_conflict: float = 0.0    # conflict degree between sources

    @property
    def interactive_elements(self) -> list[UIElement]:
        """Only elements the user/agent can interact with."""
        return [e for e in self.elements if e.clickable or e.scrollable or e.editable]

    @property
    def scrollable_elements(self) -> list[UIElement]:
        """Scrollable containers/regions on current screen."""
        return [e for e in self.elements if e.scrollable]

    def infer_viewport_hints(self) -> list[str]:
        """Heuristic hints for likely off-screen content directions."""
        hints: list[str] = []
        edge_margin = 24
        for region in self.scrollable_elements:
            left, top, right, bottom = region.bounds
            if right <= left or bottom <= top:
                continue

            children_in_region = [
                elem
                for elem in self.elements
                if elem is not region and _inside_bounds(elem.bounds, region.bounds)
            ]
            if not children_in_region:
                hints.append(
                    f"Region [{region.index}] likely has off-screen content below (empty viewport)."
                )
                continue

            touches_top = any(
                abs(elem.bounds[1] - top) <= edge_margin for elem in children_in_region
            )
            touches_bottom = any(
                abs(elem.bounds[3] - bottom) <= edge_margin for elem in children_in_region
            )
            if touches_bottom:
                hints.append(f"Region [{region.index}] likely has more content below.")
            if touches_top:
                hints.append(f"Region [{region.index}] may have content above.")
        return hints

    def to_prompt_str(self) -> str:
        """Simplified screen representation for LLM context."""
        lines = [f"Screen: {self.activity}"]
        lines.append(f"Interactive elements ({len(self.interactive_elements)}):")
        for elem in self.interactive_elements:
            lines.append(f"  {elem.to_prompt_str()}")
        if self.scrollable_elements:
            lines.append(f"Scrollable regions ({len(self.scrollable_elements)}):")
            for region in self.scrollable_elements[:3]:
                lines.append(
                    f"  [{region.index}] {region.class_name.split('.')[-1]} bounds={region.bounds}"
                )
            hints = self.infer_viewport_hints()
            if hints:
                lines.append("Viewport hints:")
                for hint in hints[:4]:
                    lines.append(f"  - {hint}")
        return "\n".join(lines)


class ScreenParserCache:
    """Cache parser result by screen hash to skip duplicate parsing."""

    def __init__(self) -> None:
        self._last_hash: str | None = None
        self._last_state: ScreenState | None = None

    def parse(self, xml_str: str) -> tuple[ScreenState, bool]:
        screen_hash = compute_screen_hash(xml_str)
        if self._last_hash == screen_hash and self._last_state is not None:
            return self._last_state, True

        state = parse_ui_hierarchy(xml_str)
        self._last_hash = screen_hash
        self._last_state = state
        return state, False


def _parse_bounds(bounds_str: str) -> tuple[int, int, int, int]:
    """Parse '[left,top][right,bottom]' format."""
    try:
        parts = bounds_str.replace("][", ",").strip("[]").split(",")
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    except (ValueError, IndexError):
        return (0, 0, 0, 0)


def _inside_bounds(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return (
        inner[0] >= outer[0]
        and inner[1] >= outer[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
    )


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"true", "1"}


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    if value.lower() in {"true", "1"}:
        return True
    if value.lower() in {"false", "0"}:
        return False
    return None


def _first_non_empty_attr(node: ET.Element, *keys: str) -> str:
    for key in keys:
        value = node.get(key)
        if value not in {None, ""}:
            return value
    return ""


def _extract_xml_payload(raw: str) -> str | None:
    start_positions = [pos for pos in (raw.find("<?xml"), raw.find("<hierarchy")) if pos != -1]
    if not start_positions:
        return None
    return raw[min(start_positions):]


def _sanitize_xml(raw: str) -> str:
    # Drop non-printable control chars that frequently break XML parser on some devices.
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", raw)


def compute_screen_hash(xml_str: str) -> str:
    """Compute deterministic hash for screen dedupe."""
    payload = _extract_xml_payload(xml_str) or xml_str
    normalized = re.sub(r"\s+", " ", _sanitize_xml(payload)).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parse_ui_hierarchy(xml_str: str) -> ScreenState:
    """Parse uiautomator XML dump into ScreenState."""
    payload = _extract_xml_payload(xml_str)
    if payload is None:
        return ScreenState(raw_xml=xml_str)

    cleaned_payload = _sanitize_xml(payload)
    try:
        root = ET.fromstring(cleaned_payload)
    except ET.ParseError:
        return ScreenState(raw_xml=cleaned_payload)

    elements: list[UIElement] = []
    idx = 0
    package_hint = _first_non_empty_attr(root, "package")

    def _walk(node: ET.Element) -> UIElement | None:
        nonlocal idx
        nonlocal package_hint
        class_name = _first_non_empty_attr(node, "class", "className")
        resource_id = _first_non_empty_attr(node, "resource-id", "resourceId")
        content_desc = _first_non_empty_attr(node, "content-desc", "contentDescription")
        text = _first_non_empty_attr(node, "text")
        if not package_hint:
            package_hint = _first_non_empty_attr(node, "package")

        elem = UIElement(
            index=idx,
            resource_id=resource_id,
            class_name=class_name,
            text=text,
            content_desc=content_desc,
            bounds=_parse_bounds(_first_non_empty_attr(node, "bounds")),
            clickable=_parse_bool(_first_non_empty_attr(node, "clickable")),
            scrollable=_parse_bool(_first_non_empty_attr(node, "scrollable")),
            editable=class_name.endswith("EditText")
            or _parse_bool(_first_non_empty_attr(node, "editable")),
            checked=_parse_optional_bool(node.get("checked")),
            enabled=_parse_bool(node.get("enabled"), default=True),
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

    package = package_hint
    if not package:
        for elem in elements:
            if ":" in elem.resource_id:
                package = elem.resource_id.split(":", maxsplit=1)[0]
                break

    return ScreenState(
        activity=package,
        package=package,
        elements=elements,
        raw_xml=cleaned_payload,
    )
