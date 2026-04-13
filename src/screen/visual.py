"""Visual understanding module — multimodal LLM integration for screenshot analysis.

Provides:
1. Screenshot analysis via vision-capable LLMs (Qwen-VL, GPT-4o, Claude)
2. Visual grounding: locate UI elements by natural language description
3. Fallback when accessibility tree is unavailable or incomplete
4. Integration with fusion pipeline via candidates_from_visual()
"""

from __future__ import annotations

import asyncio
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
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.8
    description: str = ""

    @property
    def center(self) -> tuple[int, int]:
        return (
            (self.bounds[0] + self.bounds[2]) // 2,
            (self.bounds[1] + self.bounds[3]) // 2,
        )


@dataclass
class VisualAnalysis:
    """Full visual analysis result for a screenshot."""

    elements: list[VisualElement] = field(default_factory=list)
    scene_description: str = ""
    app_name: str = ""
    raw_response: str = ""

    def to_label_scores(self) -> dict[str, float]:
        """Convert to {label: confidence} dict for fusion pipeline."""
        scores: dict[str, float] = {}
        for e in self.elements:
            label = (e.text or e.description or e.element_type).strip().lower()
            if label:
                scores[label] = max(scores.get(label, 0.0), e.confidence)
        return scores

    def find_by_text(self, query: str) -> list[VisualElement]:
        q = query.lower()
        return [e for e in self.elements if q in e.text.lower() or q in e.description.lower()]


@dataclass
class GroundingResult:
    """Result of visual grounding — locating an element by description."""

    found: bool
    x: int = 0
    y: int = 0
    bounds: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    description: str = ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYZE_PROMPT = """Analyze this Android phone screenshot. Identify ALL interactive UI elements.

Return a JSON object:
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
- Estimate bounds as accurately as possible"""

GROUNDING_PROMPT_TEMPLATE = """Look at this Android screenshot.
Find the UI element matching: "{description}"

Return JSON: {{"found": true/false, "x": center_x, "y": center_y, "bounds": [l,t,r,b], "confidence": 0.0-1.0, "description": "what you found"}}
If not visible, set "found": false."""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image_base64(image: Image.Image, max_size: int = 1024) -> str:
    """Encode PIL Image as base64 string, resizing if too large."""
    w, h = image.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        image = image.resize((int(w * ratio), int(h * ratio)))

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response text."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # First {...} block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# VisualEngine
# ---------------------------------------------------------------------------

class VisualEngine:
    """Multimodal LLM-based screen understanding engine.

    Uses any OpenAI-compatible vision API (Qwen-VL, GPT-4o, Claude).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        enabled: bool = True,
        max_image_size: int = 1024,
    ) -> None:
        self._llm = llm_client
        self.enabled = enabled
        self._max_image_size = max_image_size

    async def analyze_screen(self, screenshot: Image.Image) -> VisualAnalysis:
        """Analyze a screenshot to identify all UI elements."""
        if not self.enabled:
            return VisualAnalysis()

        b64 = encode_image_base64(screenshot, self._max_image_size)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": ANALYZE_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]

        try:
            result = await self._llm.chat(messages, temperature=0.1)
            raw = result["content"]
            return self._parse_analysis(raw)
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            return VisualAnalysis()

    async def ground(self, screenshot: Image.Image, description: str) -> GroundingResult:
        """Locate a UI element by natural language description."""
        if not self.enabled:
            return GroundingResult(found=False)

        b64 = encode_image_base64(screenshot, self._max_image_size)
        prompt = GROUNDING_PROMPT_TEMPLATE.format(description=description)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ]

        try:
            result = await self._llm.chat(messages, temperature=0.1)
            return self._parse_grounding(result["content"])
        except Exception as e:
            logger.warning(f"Visual grounding failed: {e}")
            return GroundingResult(found=False)

    @staticmethod
    def _parse_analysis(raw: str) -> VisualAnalysis:
        """Parse LLM JSON response into VisualAnalysis."""
        data = extract_json(raw)
        if not data:
            logger.warning("Failed to parse visual analysis response")
            return VisualAnalysis(raw_response=raw)

        elements = []
        for e in data.get("elements", []):
            bounds_raw = e.get("bounds", [0, 0, 0, 0])
            bounds = tuple(int(b) for b in bounds_raw) if len(bounds_raw) == 4 else (0, 0, 0, 0)
            elements.append(VisualElement(
                text=e.get("text", ""),
                element_type=e.get("type", "unknown"),
                bounds=bounds,
                confidence=float(e.get("confidence", 0.75)),
                description=e.get("description", ""),
            ))

        logger.debug(f"Visual analysis: {len(elements)} elements")
        return VisualAnalysis(
            elements=elements,
            scene_description=data.get("scene", ""),
            app_name=data.get("app", ""),
            raw_response=raw,
        )

    @staticmethod
    def _parse_grounding(raw: str) -> GroundingResult:
        data = extract_json(raw)
        if not data:
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
# Fallback pipeline (when accessibility tree is unavailable)
# ---------------------------------------------------------------------------

async def visual_fallback(
    screenshot: Image.Image,
    visual_engine: VisualEngine,
    ocr_extractor: Any | None = None,
) -> VisualAnalysis:
    """Fallback when UI hierarchy XML is empty or broken.

    Combines visual model + OCR to build best-effort screen understanding.
    """
    logger.info("Using visual fallback (no accessibility tree)")
    analysis = await visual_engine.analyze_screen(screenshot)

    if ocr_extractor is not None:
        try:
            blocks = ocr_extractor.extract(screenshot)
            existing_texts = {e.text.lower() for e in analysis.elements if e.text}
            for block in blocks:
                if block.text.lower() not in existing_texts and block.confidence > 0.5:
                    analysis.elements.append(VisualElement(
                        text=block.text,
                        element_type="text",
                        bounds=block.bounds,
                        confidence=block.confidence * 0.8,
                        description="OCR-detected text",
                    ))
        except Exception as e:
            logger.warning(f"OCR enrichment failed in fallback: {e}")

    return analysis
