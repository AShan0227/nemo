"""PhoneAgent — the main agent loop.

Observe → Think → Act → Verify cycle:
1. Capture screen state
2. Parse UI hierarchy + (optional) OCR + visual
3. Route to appropriate reasoning depth (entropy)
4. LLM decides next action
5. Safety layer validates action
6. Execute action on device
7. Verify outcome, update knowledge graph
8. Repeat until task complete or max steps
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from src.config.settings import AgentConfig
from src.device.adb import ADBController
from src.device.actions import Action, ActionType
from src.knowledge.graph import ScreenGraph
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.router import EntropyRouter
from src.safety.invariants import SafetyLayer, Verdict
from src.screen.parser import ScreenState, parse_ui_hierarchy


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # needs user confirmation


@dataclass
class StepRecord:
    """Record of a single agent step."""

    step: int
    screen_summary: str
    reasoning: str
    action: str
    params: dict[str, Any]
    success: bool
    error: str = ""


@dataclass
class TaskResult:
    status: TaskStatus
    summary: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    total_steps: int = 0


class PhoneAgent:
    """Main phone assistant agent."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.adb = ADBController(config.device)
        self.llm = LLMClient(config.llm)
        self.router = EntropyRouter(config.entropy_threshold_low, config.entropy_threshold_high)
        self.safety = SafetyLayer() if config.safety_enabled else None
        self.graph = ScreenGraph()
        self._step_count = 0

    async def connect(self) -> None:
        """Initialize device connection."""
        await self.adb.connect()
        logger.info("PhoneAgent ready")

    async def execute(self, task: str) -> TaskResult:
        """Execute a user task end-to-end."""
        logger.info(f"Task: {task}")
        self._step_count = 0
        steps: list[StepRecord] = []

        while self._step_count < self.config.max_steps:
            self._step_count += 1
            logger.info(f"--- Step {self._step_count} ---")

            # 1. Observe
            screen = await self._observe()

            # 2. Think
            screen_context = screen.to_prompt_str()
            action_json = await self.llm.decide_action(SYSTEM_PROMPT, screen_context, task)

            # 3. Parse LLM response
            try:
                decision = self._parse_decision(action_json)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                steps.append(StepRecord(
                    step=self._step_count,
                    screen_summary=screen_context[:200],
                    reasoning="parse_error",
                    action="error",
                    params={},
                    success=False,
                    error=str(e),
                ))
                continue

            reasoning = decision.get("reasoning", "")
            action_name = decision.get("action", "")
            params = decision.get("params", {})

            # Check if task is done
            if action_name == "done":
                steps.append(StepRecord(
                    step=self._step_count,
                    screen_summary=screen_context[:200],
                    reasoning=reasoning,
                    action="done",
                    params=params,
                    success=True,
                ))
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    summary=params.get("summary", reasoning),
                    steps=steps,
                    total_steps=self._step_count,
                )

            # 4. Build action
            action = self._build_action(action_name, params, screen)

            # 5. Safety check
            if self.safety and action:
                ctx = {
                    "screen_text": screen_context,
                    "step_count": self._step_count,
                    "max_steps": self.config.max_steps,
                }
                result = self.safety.check(action, ctx)
                if result.verdict == Verdict.BLOCK:
                    logger.warning(f"BLOCKED: {result.message}")
                    steps.append(StepRecord(
                        step=self._step_count,
                        screen_summary=screen_context[:200],
                        reasoning=reasoning,
                        action=action_name,
                        params=params,
                        success=False,
                        error=f"Safety: {result.message}",
                    ))
                    return TaskResult(
                        status=TaskStatus.BLOCKED,
                        summary=result.message,
                        steps=steps,
                        total_steps=self._step_count,
                    )

            # 6. Execute
            success = False
            error = ""
            if action:
                try:
                    await self._execute_action(action)
                    success = True
                except Exception as e:
                    error = str(e)
                    logger.error(f"Action failed: {e}")

            steps.append(StepRecord(
                step=self._step_count,
                screen_summary=screen_context[:200],
                reasoning=reasoning,
                action=action_name,
                params=params,
                success=success,
                error=error,
            ))

            # 7. Delay
            await asyncio.sleep(self.config.action_delay_ms / 1000)

        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Max steps ({self.config.max_steps}) exceeded",
            steps=steps,
            total_steps=self._step_count,
        )

    async def _observe(self) -> ScreenState:
        """Capture and parse current screen state."""
        xml = await self.adb.get_ui_hierarchy()
        return parse_ui_hierarchy(xml)

    def _parse_decision(self, raw: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        # Try to find JSON in the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"No JSON found in response: {raw[:200]}")

    def _build_action(self, name: str, params: dict, screen: ScreenState) -> Action | None:
        """Convert LLM decision to executable Action."""
        if name == "tap":
            idx = params.get("index", 0)
            elems = screen.interactive_elements
            if 0 <= idx < len(elems):
                x, y = elems[idx].center
                return Action.tap(x, y, f"Tap element [{idx}]")
        elif name == "type_text":
            idx = params.get("index", 0)
            text = params.get("text", "")
            elems = screen.interactive_elements
            if 0 <= idx < len(elems):
                x, y = elems[idx].center
                return Action.tap(x, y, f"Focus [{idx}] then type")
                # Note: typing follows after tap, handled in _execute_action
        elif name == "scroll":
            direction = params.get("direction", "down")
            return Action.scroll(direction)
        elif name == "back":
            return Action.back()
        elif name == "home":
            return Action.home()
        elif name == "launch":
            package = params.get("package", "")
            return Action.launch_app(package)
        elif name == "wait":
            ms = params.get("ms", 1000)
            return Action.wait(ms)
        return None

    async def _execute_action(self, action: Action) -> None:
        """Execute an action on the device."""
        match action.type:
            case ActionType.TAP:
                await self.adb.tap(action.params["x"], action.params["y"])
            case ActionType.LONG_PRESS:
                await self.adb.long_press(
                    action.params["x"], action.params["y"],
                    action.params.get("duration_ms", 1000),
                )
            case ActionType.DOUBLE_TAP:
                await self.adb.double_tap(action.params["x"], action.params["y"])
            case ActionType.SWIPE:
                await self.adb.swipe(
                    action.params["x1"], action.params["y1"],
                    action.params["x2"], action.params["y2"],
                    action.params.get("duration_ms", 300),
                )
            case ActionType.TYPE_TEXT:
                await self.adb.input_text(action.params["text"])
            case ActionType.PRESS_KEY:
                await self.adb.press_key(action.params["keycode"])
            case ActionType.BACK:
                await self.adb.press_back()
            case ActionType.HOME:
                await self.adb.press_home()
            case ActionType.LAUNCH_APP:
                await self.adb.launch_app(action.params["package"])
            case ActionType.SCROLL:
                direction = action.params.get("direction", "down")
                sw, sh = self.adb.screen_size
                cx, cy = sw // 2, sh // 2
                dy = sh // 4 if direction == "down" else -(sh // 4)
                await self.adb.swipe(cx, cy, cx, cy - dy, 300)
            case ActionType.PINCH_ZOOM:
                await self.adb.pinch_zoom(
                    action.params["cx"], action.params["cy"],
                    action.params.get("zoom_in", True),
                    action.params.get("scale", 0.5),
                )
            case ActionType.WAIT:
                await asyncio.sleep(action.params.get("ms", 1000) / 1000)
