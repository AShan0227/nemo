"""PhoneAgent — the main agent loop.

Observe → Think → Act → Verify cycle:
1. Capture screen state
2. Parse UI hierarchy + (optional) OCR + visual fusion
3. Route to appropriate reasoning depth (entropy)
4. LLM decides next action (or use cached/planned action)
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

from src.agent.planner import TaskPlanner
from src.config.settings import AgentConfig
from src.device.adb import ADBController
from src.device.actions import Action, ActionType
from src.knowledge.graph import ScreenGraph, ScreenNode
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.router import EntropyRouter, ReasoningDepth
from src.safety.invariants import SafetyLayer, Verdict
from src.screen.fusion import EvidenceMass, fuse_multiple
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
        self.planner = TaskPlanner(self.graph)
        self._step_count = 0
        self._screen_size: tuple[int, int] = (1080, 1920)  # updated on connect

    async def connect(self) -> None:
        """Initialize device connection."""
        await self.adb.connect()
        self._screen_size = await self.adb.get_screen_size()
        logger.info(f"PhoneAgent ready (screen: {self._screen_size[0]}x{self._screen_size[1]})")

    async def execute(self, task: str) -> TaskResult:
        """Execute a user task end-to-end."""
        logger.info(f"Task: {task}")
        self._step_count = 0
        steps: list[StepRecord] = []
        prev_screen_hash: str | None = None

        while self._step_count < self.config.max_steps:
            self._step_count += 1
            logger.info(f"--- Step {self._step_count} ---")

            # 1. Observe (with fusion framework)
            screen = await self._observe()
            screen_hash = ScreenGraph.compute_screen_hash(screen.raw_xml)

            # Update graph node visit count
            if screen_hash not in self.graph._nodes:
                self.graph.add_node(ScreenNode(
                    state_id=screen_hash,
                    activity=screen.activity,
                    package=screen.package,
                    ui_hash=screen_hash,
                ))
            self.graph._nodes[screen_hash].visit_count += 1

            # 2. Route — decide reasoning depth
            screen_context = screen.to_prompt_str()
            routing_decision = self._route_decision(screen_hash)

            action_json: str | None = None
            used_cache = False

            if routing_decision.depth == ReasoningDepth.SYSTEM_1:
                cached = self.router.get_cached_action(screen_hash)
                if cached:
                    action_json = cached
                    used_cache = True
                    logger.info("System 1: using cached action")

            # 3. Think — LLM decision (if not cached)
            if action_json is None:
                action_json = await self.llm.decide_action(SYSTEM_PROMPT, screen_context, task)

            # 4. Parse LLM response
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

            # 5. Build action
            action = self._build_action(action_name, params, screen)

            # 6. Safety check
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

            # 7. Execute
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

            # 8. Update knowledge graph with transition
            if action and prev_screen_hash:
                new_screen = await self._observe()
                new_hash = ScreenGraph.compute_screen_hash(new_screen.raw_xml)
                self.graph.record_transition(
                    prev_screen_hash, new_hash, action_name, success,
                )
                # Cache successful actions for System 1
                if success and not used_cache:
                    self.router.cache_action(prev_screen_hash, action_json)
            else:
                new_hash = screen_hash

            prev_screen_hash = new_hash

            # Periodic pheromone evaporation
            if self._step_count % 10 == 0:
                self.graph.evaporate_pheromones()

            # 9. Delay
            await asyncio.sleep(self.config.action_delay_ms / 1000)

        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Max steps ({self.config.max_steps}) exceeded",
            steps=steps,
            total_steps=self._step_count,
        )

    async def _observe(self) -> ScreenState:
        """Capture and parse current screen state with fusion framework."""
        xml = await self.adb.get_ui_hierarchy()
        screen = parse_ui_hierarchy(xml)

        if self.config.fusion_enabled:
            # Source 1: UI hierarchy evidence
            ui_evidence = self._build_ui_evidence(screen)
            # Source 2: OCR (stub — Surf will integrate)
            ocr_evidence = EvidenceMass({"unknown": 1.0})
            # Source 3: Visual model (stub — Surf will integrate)
            visual_evidence = EvidenceMass({"unknown": 1.0})

            fused, conflict = fuse_multiple(ui_evidence, ocr_evidence, visual_evidence)
            screen.fusion_confidence = 1.0 - conflict
            screen.fusion_conflict = conflict

        return screen

    def _build_ui_evidence(self, screen: ScreenState) -> EvidenceMass:
        """Create evidence mass from UI hierarchy parse result."""
        n_interactive = len(screen.interactive_elements)
        n_total = len(screen.elements)
        if n_total == 0:
            return EvidenceMass({"unknown": 1.0})
        ratio = n_interactive / n_total
        return EvidenceMass({
            "interactive_screen": ratio,
            "static_screen": 1.0 - ratio,
        })

    def _route_decision(self, screen_hash: str):
        """Decide reasoning depth based on graph knowledge (entropy proxy)."""
        node = self.graph._nodes.get(screen_hash)
        if node is None:
            # Unknown screen — full reasoning
            return self.router.route(screen_hash, [1.0])

        edges = self.graph.get_neighbors(screen_hash)
        if not edges:
            # Known screen but no known actions
            return self.router.route(screen_hash, [0.5, 0.5])

        # Compute proxy entropy from edge success rates
        rates = [max(e.success_rate, 0.01) for e in edges]
        total = sum(rates)
        probs = [r / total for r in rates]
        return self.router.route(screen_hash, probs)

    def _parse_decision(self, raw: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
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
                return Action(
                    ActionType.TYPE_TEXT,
                    {"x": x, "y": y, "text": text},
                    f"Type '{text}' at [{idx}]",
                )
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
        elif name == "done":
            return None  # handled in execute() before builder
        return None

    async def _execute_action(self, action: Action) -> None:
        """Execute an action on the device."""
        t = action.type
        if t == ActionType.TAP:
            await self.adb.tap(action.params["x"], action.params["y"])
        elif t == ActionType.SWIPE:
            await self.adb.swipe(
                action.params["x1"], action.params["y1"],
                action.params["x2"], action.params["y2"],
                action.params.get("duration_ms", 300),
            )
        elif t == ActionType.TYPE_TEXT:
            if "x" in action.params and "y" in action.params:
                await self.adb.tap(action.params["x"], action.params["y"])
                await asyncio.sleep(0.3)
            await self.adb.input_text(action.params["text"])
        elif t == ActionType.PRESS_KEY:
            await self.adb.press_key(action.params["keycode"])
        elif t == ActionType.BACK:
            await self.adb.press_back()
        elif t == ActionType.HOME:
            await self.adb.press_home()
        elif t == ActionType.LAUNCH_APP:
            await self.adb.launch_app(action.params["package"])
        elif t == ActionType.SCROLL:
            direction = action.params.get("direction", "down")
            w, h = self._screen_size
            cx, cy = w // 2, h // 2
            dy = h // 4 if direction == "down" else -(h // 4)
            await self.adb.swipe(cx, cy, cx, cy - dy, 300)
        elif t == ActionType.WAIT:
            await asyncio.sleep(action.params.get("ms", 1000) / 1000)
