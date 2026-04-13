"""PhoneAgent — the main agent loop.

Observe → Think → Act → Verify cycle:
1. Capture screen state (UI hierarchy + OCR + visual fusion)
2. Route to appropriate reasoning depth (entropy)
3. LLM decides next action (or use cached/planned action)
4. Safety layer validates action
5. Execute action on device (with retry + screenshot verify)
6. Update knowledge graph with transition
7. Repeat until task complete or max steps
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from src.agent.planner import TaskPlanner
from src.config.settings import AgentConfig
from src.device.actions import Action, ActionRecorder, ActionTimingTracker, ActionType
from src.device.adb import ADBController
from src.device.replay import ActionReplayer, ReplayReport
from src.knowledge.graph import ScreenGraph, ScreenNode
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT
from src.llm.router import EntropyRouter, ReasoningDepth
from src.safety.invariants import SafetyLayer, Verdict
from src.screen.fusion import candidates_from_ocr, candidates_from_ui, fuse_screen_sources
from src.screen.ocr import OCRExtractor
from src.screen.parser import ScreenParserCache, ScreenState


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class StepRecord:
    """Record of a single agent step."""

    step: int
    screen_summary: str
    reasoning: str
    action: str
    params: dict[str, Any]
    success: bool
    duration_ms: float = 0.0
    attempts: int = 1
    error: str = ""


@dataclass
class TaskResult:
    status: TaskStatus
    summary: str = ""
    steps: list[StepRecord] = field(default_factory=list)
    total_steps: int = 0
    timing_stats: dict[str, dict[str, float]] = field(default_factory=dict)


class PhoneAgent:
    """Main phone assistant agent."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.adb = ADBController(config.device)
        self.llm = LLMClient(config.llm)
        self.router = EntropyRouter(config.entropy_threshold_low, config.entropy_threshold_high)
        if config.safety_enabled:
            if config.safety_rules_path:
                self.safety = SafetyLayer.from_yaml(config.safety_rules_path)
            else:
                self.safety = SafetyLayer(audit_log_path=config.safety_audit_log_path)
        else:
            self.safety = None
        self.graph = ScreenGraph()
        self.planner = TaskPlanner(self.graph)
        self.screen_parser = ScreenParserCache()
        self.ocr = OCRExtractor(
            enabled=config.ocr_enabled,
            min_confidence=config.ocr_min_confidence,
        )
        self.action_recorder = ActionRecorder()
        self.action_timing = ActionTimingTracker()
        self._step_count = 0
        self._screen_size: tuple[int, int] = (1080, 1920)

    async def connect(self) -> None:
        """Initialize device connection."""
        await self.adb.connect()
        self._screen_size = await self.adb.get_screen_size()
        self._load_graph_from_disk()
        logger.info(f"PhoneAgent ready (screen: {self._screen_size[0]}x{self._screen_size[1]})")

    async def replay_recording(
        self,
        path: str | Path,
        *,
        speed: float = 1.0,
        preserve_timing: bool = True,
        stop_on_error: bool = True,
    ) -> ReplayReport:
        """Replay a previously recorded action trace."""
        replayer = ActionReplayer(self._execute_action)
        return await replayer.replay_file(
            path, speed=speed, preserve_timing=preserve_timing, stop_on_error=stop_on_error,
        )

    async def execute(self, task: str) -> TaskResult:
        """Execute a user task end-to-end."""
        logger.info(f"Task: {task}")
        self._step_count = 0
        steps: list[StepRecord] = []
        prev_screen_hash: str | None = None
        active_plan = None
        plan_cursor = 0
        self.action_recorder.clear()
        self.action_timing = ActionTimingTracker()

        while self._step_count < self.config.max_steps:
            self._step_count += 1
            logger.info(f"--- Step {self._step_count} ---")

            # 1. Observe
            screen = await self._observe()
            screen_hash = ScreenGraph.compute_screen_hash(screen.raw_xml)

            # Update graph node visit count
            if screen_hash not in self.graph._nodes:
                self.graph.add_node(ScreenNode(
                    state_id=screen_hash, activity=screen.activity,
                    package=screen.package, ui_hash=screen_hash,
                ))
            self.graph._nodes[screen_hash].visit_count += 1

            # 2. Route — decide reasoning depth
            screen_context = await self._build_screen_context(screen)
            routing_decision = self._route_decision(screen_hash)

            if self.config.planner_enabled and active_plan is None:
                planner_chat = self._resolve_llm_method("chat")
                if planner_chat is not None:
                    try:
                        active_plan = await self.planner.plan_with_llm(
                            self.llm,
                            task=task,
                            current_app=screen.activity,
                            screen_summary=screen_context,
                        )
                    except Exception as exc:
                        logger.debug("Planner fallback unavailable: {}", exc)

            action_json: str | None = None
            decision: dict[str, Any] | None = None
            used_cache = False

            if routing_decision.depth == ReasoningDepth.SYSTEM_1:
                cached = self.router.get_cached_action(screen_hash)
                if cached:
                    action_json = cached
                    used_cache = True
                    logger.info("System 1: using cached action")

            # 3. Think — LLM decision (if not cached)
            if action_json is None:
                planned_hint = self.planner.next_action_hint(active_plan, plan_cursor)
                if (
                    self.config.planner_enabled
                    and planned_hint is not None
                    and routing_decision.depth != ReasoningDepth.SYSTEM_2
                ):
                    decision = {
                        "reasoning": "planner_fallback",
                        "action": planned_hint.action_type,
                        "params": planned_hint.params,
                    }
                    plan_cursor += 1
                    logger.info(
                        "Using planner hint step {} -> {}",
                        planned_hint.step_number,
                        planned_hint.action_type,
                    )
                else:
                    structured_method = self._resolve_llm_method("decide_action_structured")
                    if structured_method is not None:
                        try:
                            structured_payload = await structured_method(
                                SYSTEM_PROMPT,
                                screen_context,
                                task,
                            )
                            if isinstance(structured_payload, dict):
                                decision = structured_payload
                            elif isinstance(structured_payload, str):
                                action_json = structured_payload
                            else:
                                logger.debug(
                                    "Structured decision ignored due to payload type: {}",
                                    type(structured_payload).__name__,
                                )
                        except Exception as exc:
                            logger.debug("Structured decision unavailable: {}", exc)

                    if decision is None and action_json is None:
                        decide_method = self._resolve_llm_method("decide_action")
                        if decide_method is None:
                            raise RuntimeError("LLM client missing `decide_action` interface")
                        legacy_payload = await decide_method(
                            SYSTEM_PROMPT,
                            screen_context,
                            task,
                        )
                        if isinstance(legacy_payload, dict):
                            decision = legacy_payload
                        else:
                            action_json = str(legacy_payload)

            # 4. Parse LLM response
            if decision is None:
                try:
                    decision = self._parse_decision(action_json or "")
                except Exception as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    steps.append(StepRecord(
                        step=self._step_count, screen_summary=screen_context[:200],
                        reasoning="parse_error", action="error", params={},
                        success=False, error=str(e),
                    ))
                    continue

            reasoning = decision.get("reasoning", "")
            action_name = decision.get("action", "")
            params = decision.get("params", {})

            # Check if task is done
            if action_name == "done":
                steps.append(StepRecord(
                    step=self._step_count, screen_summary=screen_context[:200],
                    reasoning=reasoning, action="done", params=params, success=True,
                ))
                result = TaskResult(
                    status=TaskStatus.COMPLETED, summary=params.get("summary", reasoning),
                    steps=steps, total_steps=self._step_count,
                    timing_stats=self.action_timing.snapshot(),
                )
                self._save_graph_to_disk()
                return result

            # 5. Build action
            action = self._build_action(action_name, params, screen)
            if action is None:
                steps.append(StepRecord(
                    step=self._step_count, screen_summary=screen_context[:200],
                    reasoning=reasoning, action=action_name, params=params,
                    success=False, error=f"Unknown or invalid action: {action_name}",
                ))
                continue

            # 6. Safety check
            if self.safety:
                ctx = {
                    "screen_text": screen_context,
                    "step_count": self._step_count,
                    "max_steps": self.config.max_steps,
                }
                result = self.safety.check(action, ctx)
                if result.verdict == Verdict.BLOCK:
                    logger.warning(f"BLOCKED: {result.message}")
                    steps.append(StepRecord(
                        step=self._step_count, screen_summary=screen_context[:200],
                        reasoning=reasoning, action=action_name, params=params,
                        success=False, error=f"Safety: {result.message}",
                    ))
                    blocked_result = TaskResult(
                        status=TaskStatus.BLOCKED, summary=result.message,
                        steps=steps, total_steps=self._step_count,
                        timing_stats=self.action_timing.snapshot(),
                    )
                    self._save_graph_to_disk()
                    return blocked_result

            # 7. Execute with retry + verify
            success, error, duration_ms, attempts = await self._run_action_with_retry(action)
            self.action_timing.record(action.type, duration_ms, success)
            self.action_recorder.record(
                timestamp_ms=int(time.time() * 1000), action=action,
                success=success, error=error, duration_ms=duration_ms, attempts=attempts,
            )

            steps.append(StepRecord(
                step=self._step_count, screen_summary=screen_context[:200],
                reasoning=reasoning, action=action_name, params=params,
                success=success, duration_ms=duration_ms, attempts=attempts, error=error,
            ))

            # 8. Update knowledge graph with transition
            if prev_screen_hash:
                new_screen = await self._observe()
                new_hash = ScreenGraph.compute_screen_hash(new_screen.raw_xml)
                self.graph.record_transition(prev_screen_hash, new_hash, action_name, success)
                if success and not used_cache:
                    cache_payload = action_json or json.dumps(decision, ensure_ascii=False)
                    self.router.cache_action(prev_screen_hash, cache_payload)
            else:
                new_hash = screen_hash

            prev_screen_hash = new_hash

            # Periodic pheromone evaporation
            if self._step_count % 10 == 0:
                self.graph.evaporate_pheromones()

            # 9. Delay
            await asyncio.sleep(self.config.action_delay_ms / 1000)

        failed_result = TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Max steps ({self.config.max_steps}) exceeded",
            steps=steps, total_steps=self._step_count,
            timing_stats=self.action_timing.snapshot(),
        )
        self._save_graph_to_disk()
        return failed_result

    def _load_graph_from_disk(self) -> None:
        path_str = getattr(self.config, "graph_persist_path", None)
        if not path_str:
            return
        path = Path(path_str)
        if not path.exists():
            return
        try:
            self.graph = ScreenGraph.load(path)
            self.planner = TaskPlanner(self.graph)
            logger.info("Loaded graph from {}", path)
        except Exception as exc:
            logger.warning("Failed to load graph from {}: {}", path, exc)

    def _save_graph_to_disk(self) -> None:
        path_str = getattr(self.config, "graph_persist_path", None)
        if not path_str:
            return
        path = Path(path_str)
        try:
            self.graph.save(path)
        except Exception as exc:
            logger.warning("Failed to save graph to {}: {}", path, exc)

    async def _run_action_with_retry(self, action: Action) -> tuple[bool, str, float, int]:
        """Execute action with step-level retry and optional visual verification."""
        max_attempts = max(1, self.config.step_retry_count + 1)
        started = time.perf_counter()
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            try:
                before_path: Path | None = None
                if self._should_verify(action):
                    before_path = await self.adb.save_screenshot()

                await self._execute_action(action)

                if before_path is not None:
                    after_path = await self.adb.save_screenshot()
                    changed = await self.adb.screenshot_changed(
                        before_path, after_path, threshold=self.config.verify_diff_threshold,
                    )
                    if not changed:
                        raise RuntimeError("verify_failed: screen appears unchanged")

                total_duration = (time.perf_counter() - started) * 1000
                return True, "", total_duration, attempt
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Action {} failed attempt {}/{}: {}",
                    action.type.value, attempt, max_attempts, last_error,
                )
                if attempt >= max_attempts:
                    break
                await asyncio.sleep(0.2)

        total_duration = (time.perf_counter() - started) * 1000
        return False, last_error, total_duration, max_attempts

    def _should_verify(self, action: Action) -> bool:
        if not self.config.verify_enabled:
            return False
        return action.type != ActionType.WAIT

    async def _observe(self) -> ScreenState:
        """Capture and parse current screen state."""
        xml = await self.adb.get_ui_hierarchy()
        state, dedup_hit = self.screen_parser.parse(xml)
        if dedup_hit:
            logger.debug("Screen hash unchanged, reusing last parsed state")
        return state

    async def _build_screen_context(self, screen: ScreenState) -> str:
        """Build LLM-visible screen context with optional OCR/fusion hints."""
        context = screen.to_prompt_str()
        if not self.config.ocr_enabled:
            return context

        try:
            screenshot = await self.adb.screenshot()
            ocr_blocks = await asyncio.to_thread(self.ocr.extract, screenshot)
        except Exception as exc:
            logger.warning(f"Failed to run OCR during observe: {exc}")
            return context

        if not ocr_blocks:
            return context

        fusion = fuse_screen_sources(
            ui_candidates=candidates_from_ui(screen.interactive_elements),
            ocr_candidates=candidates_from_ocr(
                ocr_blocks, min_confidence=self.config.ocr_min_confidence,
            ),
            top_k=self.config.fusion_top_k,
        )

        lines = [context, "", f"OCR lines ({len(ocr_blocks)}):"]
        for block in ocr_blocks[:8]:
            lines.append(f'  - "{block.text}" ({block.confidence:.2f})')
        lines.append(f"Fused hypotheses (conflict={fusion.conflict:.2f}):")
        for label, score in fusion.ranked[:self.config.fusion_top_k]:
            lines.append(f"  - {label}: {score:.2f}")
        return "\n".join(lines)

    def _route_decision(self, screen_hash: str):
        """Decide reasoning depth based on graph knowledge (entropy proxy)."""
        node = self.graph._nodes.get(screen_hash)
        if node is None:
            return self.router.route(screen_hash, [1.0])
        edges = self.graph.get_neighbors(screen_hash)
        if not edges:
            return self.router.route(screen_hash, [0.5, 0.5])
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

    def _resolve_llm_method(self, name: str) -> Callable[..., Any] | None:
        """Return a callable LLM method when explicitly present."""
        instance_dict = getattr(self.llm, "__dict__", {})
        has_instance_attr = isinstance(instance_dict, dict) and name in instance_dict
        has_class_attr = hasattr(type(self.llm), name)
        if not (has_instance_attr or has_class_attr):
            return None
        method = getattr(self.llm, name, None)
        return method if callable(method) else None

    def _build_action(self, name: str, params: dict, screen: ScreenState) -> Action | None:
        """Convert LLM decision to executable Action."""
        elems = screen.interactive_elements

        def _resolve_center(index: int) -> tuple[int, int] | None:
            if 0 <= index < len(elems):
                return elems[index].center
            return None

        if name == "tap":
            center = _resolve_center(params.get("index", 0))
            if center:
                return Action.tap(center[0], center[1], f"Tap element [{params.get('index', 0)}]")
        elif name == "type_text":
            idx = params.get("index", 0)
            text = params.get("text", "")
            center = _resolve_center(idx)
            if center:
                return Action(
                    ActionType.TYPE_TEXT,
                    {"x": center[0], "y": center[1], "text": text},
                    f"Type '{text}' at [{idx}]",
                )
        elif name == "scroll":
            return Action.scroll(params.get("direction", "down"))
        elif name == "back":
            return Action.back()
        elif name == "home":
            return Action.home()
        elif name == "launch":
            return Action.launch_app(params.get("package", ""))
        elif name == "wait":
            return Action.wait(params.get("ms", 1000))
        elif name == "done":
            return None
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
