"""Visual understanding module — multimodal LLM integration for screenshot analysis.

Provides:
1. Screenshot analysis via vision-capable LLMs (Qwen-VL, GPT-4o, Claude)
2. Visual grounding: locate UI elements by natural language description
3. Fallback when accessibility tree is unavailable or incomplete
"""

from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image

    from src.llm.client import LLMClient


@dataclass
class VisualElement:
    """A UI element identified by the visual model."""

    text: str
    element_type: str  # button, icon, text, input, image, link, etc.
    bounds: tuple[int, int, int, int]  # left, top, right, bottom
    confidence: float = 0.8
    description: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )

    def to_fusion_dict(self) -> dict[str, Any]:
        """Convert to format expected by fuse_screen()."""
        return {
            "text": self.text,
            "bounds": self.bounds,
            "type": self.element_type,
            "confidence": self.confidence,
        }


@dataclass
class VisualAnalysis:
    """Full visual analysis result for a screenshot."""

    elements: list[VisualElement] = field(default_factory=list)
    scene_description: str = ""
    app_name: str = ""
    raw_response: str = ""

    def find_by_text(self, query: str) -> list[VisualElement]:
        q = query.lower()
        return [e for e in self.elements if q in e.text.lower() or q in e.description.lower()]

    def find_by_type(self, element_type: str) -> list[VisualElement]:
        return [e for e in self.elements if e.element_type == element_type]


@dataclass
class GroundingResult:
    """Result of visual grounding — locating an element by description."""

    found: bool
    x: int = 0
    y: int = 0
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    description: str = ""


def _encode_image_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Encode PIL Image as base64 string, resizing if too large."""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYZE_PROMPT = """Analyze this Android phone screenshot. Identify ALL interactive UI elements.

Return a JSON object with this exact format:
{
  "scene": "brief description of what's on screen",
  "app": "app name if identifiable",
  "elements": [
    {
      "text": "visible text on the element",
      "type": "button|icon|text|input|image|link|checkbox|switch|tab",
      "bounds": [left, top, right, bottom],
      "description": "what this element does"
    }
  ]
}

Rules:
- bounds are pixel coordinates [left, top, right, bottom]
- Include ALL tappable elements, icons, text fields, buttons
- For icons without text, describe what the icon looks like in "text"
- Estimate bounds as accurately as possible based on visual position"""

GROUNDING_PROMPT_TEMPLATE = """Look at this Android screenshot.
Find the UI element matching this description: "{description}"

Return a JSON object:
{{
  "found": true/false,
  "x": center_x_pixel,
  "y": center_y_pixel,
  "bounds": [left, top, right, bottom],
  "confidence": 0.0-1.0,
  "description": "what you found"
}}

If the element is not visible on screen, set "found": false."""


# ---------------------------------------------------------------------------
# VisualEngine
# ---------------------------------------------------------------------------

class VisualEngine:
    """Multimodal LLM-based screen understanding engine.

    Supports any OpenAI-compatible vision API (GPT-4o, Qwen-VL, etc.)
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def analyze_screen(self, screenshot: Image.Image) -> VisualAnalysis:
        """Analyze a screenshot to identify all UI elements.

        Args:
            screenshot: PIL Image of the phone screen.

        Returns:
            VisualAnalysis with all identified elements.
        """
        b64 = _encode_image_base64(screenshot)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYZE_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]

        result = await self._llm.chat(messages, temperature=0.1)
        raw = result["content"]

        return self._parse_analysis(raw)

    async def ground(self, screenshot: Image.Image, description: str) -> GroundingResult:
        """Locate a UI element by natural language description.

        Args:
            screenshot: PIL Image of the phone screen.
            description: What to find, e.g. "the search bar" or "微信 icon".

        Returns:
            GroundingResult with coordinates if found.
        """
        b64 = _encode_image_base64(screenshot)
        prompt = GROUNDING_PROMPT_TEMPLATE.format(description=description)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ]

        result = await self._llm.chat(messages, temperature=0.1)
        raw = result["content"]

        return self._parse_grounding(raw)

    @staticmethod
    def _parse_analysis(raw: str) -> VisualAnalysis:
        """Parse LLM JSON response into VisualAnalysis."""
        data = _extract_json(raw)
        if not data:
            logger.warning("Failed to parse visual analysis response")
            return VisualAnalysis(raw_response=raw)

        elements = []
        for e in data.get("elements", []):
            bounds_raw = e.get("bounds", [0, 0, 0, 0])
            if len(bounds_raw) == 4:
                bounds = (int(bounds_raw[0]), int(bounds_raw[1]),
                          int(bounds_raw[2]), int(bounds_raw[3]))
            else:
                bounds = (0, 0, 0, 0)

            elements.append(VisualElement(
                text=e.get("text", ""),
                element_type=e.get("type", "unknown"),
                bounds=bounds,
                description=e.get("description", ""),
            ))

        logger.debug(f"Visual analysis: {len(elements)} elements, scene={data.get('scene', '')[:60]}")
        return VisualAnalysis(
            elements=elements,
            scene_description=data.get("scene", ""),
            app_name=data.get("app", ""),
            raw_response=raw,
        )

    @staticmethod
    def _parse_grounding(raw: str) -> GroundingResult:
        """Parse LLM JSON response into GroundingResult."""
        data = _extract_json(raw)
        if not data:
            logger.warning("Failed to parse grounding response")
            return GroundingResult(found=False)

        bounds_raw = data.get("bounds", [0, 0, 0, 0])
        bounds = tuple(int(b) for b in bounds_raw) if len(bounds_raw) == 4 else (0, 0, 0, 0)

        return GroundingResult(
            found=data.get("found", False),
            x=int(data.get("x", 0)),
            y=int(data.get("y", 0)),
            bounds=bounds,
            confidence=float(data.get("confidence", 0.0)),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Fallback screen understanding (when accessibility tree is unavailable)
# ---------------------------------------------------------------------------

async def visual_fallback(
    screenshot: Image.Image,
    visual_engine: VisualEngine,
    ocr_engine: Any | None = None,
) -> VisualAnalysis:
    """Fallback pipeline when UI hierarchy XML is unavailable.

    Combines visual model analysis with OCR to build a best-effort
    screen understanding without accessibility tree.

    Args:
        screenshot: Current screen capture.
        visual_engine: Initialized VisualEngine.
        ocr_engine: Optional OCREngine for additional text extraction.

    Returns:
        VisualAnalysis with elements from vision + OCR.
    """
    logger.info("Using visual fallback (no accessibility tree)")

    # Primary: visual model analysis
    analysis = await visual_engine.analyze_screen(screenshot)

    # Secondary: OCR enrichment
    if ocr_engine is not None:
        try:
            ocr_output = await ocr_engine.recognize(screenshot)
            # Add OCR-discovered text not already in visual elements
            for ocr_r in ocr_output.results:
                already_found = any(
                    _bbox_iou(
                        (ocr_r.bbox[0], ocr_r.bbox[1], ocr_r.bbox[2], ocr_r.bbox[3]),
                        ve.bounds,
                    ) > 0.3
                    for ve in analysis.elements
                )
                if not already_found and ocr_r.confidence > 0.7:
                    analysis.elements.append(VisualElement(
                        text=ocr_r.text,
                        element_type="text",
                        bounds=ocr_r.bbox,
                        confidence=ocr_r.confidence * 0.8,
                        description="OCR-detected text",
                    ))
            logger.debug(f"Visual fallback: {len(analysis.elements)} elements after OCR enrichment")
        except Exception as e:
            logger.warning(f"OCR enrichment failed in fallback: {e}")

    return analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding JSON in markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first {...} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU between two bounding boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
