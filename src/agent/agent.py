"""PhoneAgent — the main agent loop.

Observe → Think → Act → Verify → Learn cycle with integrated research mechanisms:
1. Capture screen state (UI hierarchy + OCR + visual fusion)
2. Immune check: detect anomalous screens (crash, captcha, wrong app)
3. Homeostasis: adjust behavior based on performance metrics
4. Route to appropriate reasoning depth (entropy router)
5. LLM decides next action (or use cached/planned action)
6. Inertia: check plan stability before accepting deviation
7. Safety layer validates action
8. Execute action on device (with retry + screenshot verify)
9. Explorer: update Boltzmann Q-values for state-action pairs
10. Update knowledge graph with transition
11. Phase detector: check for capability jumps
12. Repeat until task complete or max steps
"""

from __future__ import annotations

import asyncio
import json
import time
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
from src.research.circadian import CircadianModel
from src.research.explorer import BoltzmannExplorer
from src.research.genome import Codon, Gene, Opcode, Workflow
from src.research.homeostasis import HomeostasisRegulator
from src.research.immune import ImmuneSystem, extract_features
from src.research.inertia import InertiaController
from src.research.phase_detector import PerformanceSnapshot, PhaseDetector
from src.safety.invariants import SafetyLayer, Verdict
from src.screen.fusion import candidates_from_ocr, candidates_from_ui, fuse_screen_sources
from src.screen.ocr import OCRExtractor
from src.screen.parser import ScreenParserCache, ScreenState, parse_ui_hierarchy


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

        # Research mechanisms
        self.homeostasis = HomeostasisRegulator() if config.homeostasis_enabled else None
        self.immune = ImmuneSystem(anomaly_threshold=config.immune_anomaly_threshold) if config.immune_enabled else None
        self.inertia = InertiaController(base_inertia=config.inertia_base) if config.inertia_enabled else None
        self.explorer = BoltzmannExplorer(initial_temperature=config.explorer_temperature) if config.explorer_enabled else None
        self.phase_detector = PhaseDetector() if config.phase_detector_enabled else None
        self.circadian = CircadianModel() if config.circadian_enabled else None

        self._step_count = 0
        self._screen_size: tuple[int, int] = (1080, 1920)
        self._task_start_time: float = 0.0
        self._success_window: list[bool] = []

    async def connect(self) -> None:
        """Initialize device connection and load persisted graph."""
        await self.adb.connect()
        self._screen_size = await self.adb.get_screen_size()
        # Load persisted knowledge graph
        if self.config.graph_persist_path:
            self.graph = ScreenGraph.load(self.config.graph_persist_path)
            self.planner = TaskPlanner(self.graph)
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
        self._task_start_time = time.time()
        steps: list[StepRecord] = []
        prev_screen_hash: str | None = None
        self.action_recorder.clear()
        self.action_timing = ActionTimingTracker()

        while self._step_count < self.config.max_steps:
            self._step_count += 1
            logger.info(f"--- Step {self._step_count} ---")
            step_start = time.perf_counter()

            # 1. Observe
            screen = await self._observe()
            screen_hash = ScreenGraph.compute_screen_hash(screen.raw_xml)

            # Update graph node
            if screen_hash not in self.graph._nodes:
                self.graph.add_node(ScreenNode(
                    state_id=screen_hash, activity=screen.activity,
                    package=screen.package, ui_hash=screen_hash,
                ))
            self.graph._nodes[screen_hash].visit_count += 1

            # 2. Immune check — detect anomalous screens
            if self.immune and self.immune._trained:
                features = extract_features(screen)
                anomaly = self.immune.check(features)
                if anomaly.is_anomaly:
                    logger.warning(f"Anomaly detected: {anomaly.message}. Pressing back to recover.")
                    await self.adb.press_back()
                    await asyncio.sleep(0.5)
                    continue
            elif self.immune and not self.immune._trained:
                # Still in training phase — collect self samples
                self.immune.add_self_sample(extract_features(screen))
                if len(self.immune._self_set) >= 20:
                    self.immune.train()

            # 3. Homeostasis — check if behavior adjustments needed
            if self.homeostasis:
                adjustments = self.homeostasis.get_adjustments()
                for adj in adjustments:
                    if adj.name == "increase_delay" and adj.value > 0.1:
                        self.config.action_delay_ms = min(2000, self.config.action_delay_ms + 100)

            # 3b. Circadian — apply time-of-day behavior modifiers
            if self.circadian:
                mods = self.circadian.get_behavior_modifier()
                # Scale action delay by time-of-day multiplier (e.g. slower at night)
                effective_delay = int(self.config.action_delay_ms * mods["action_delay_multiplier"])
                self.config.action_delay_ms = min(3000, effective_delay)

            # 4. Route — decide reasoning depth
            screen_context = await self._build_screen_context(screen)
            routing_decision = self._route_decision(screen_hash)

            action_json: str | None = None
            used_cache = False

            if routing_decision.depth == ReasoningDepth.SYSTEM_1:
                cached = self.router.get_cached_action(screen_hash)
                if cached:
                    action_json = cached
                    used_cache = True
                    logger.info("System 1: using cached action")

            # 5. Think — LLM decision (if not cached)
            if action_json is None:
                action_json = await self.llm.decide_action(SYSTEM_PROMPT, screen_context, task)

            # 6. Parse LLM response
            try:
                decision = self._parse_decision(action_json)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                steps.append(StepRecord(
                    step=self._step_count, screen_summary=screen_context[:200],
                    reasoning="parse_error", action="error", params={},
                    success=False, error=str(e),
                ))
                self._success_window.append(False)
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
                if self.inertia:
                    self.inertia.clear_plan()
                result = TaskResult(
                    status=TaskStatus.COMPLETED, summary=params.get("summary", reasoning),
                    steps=steps, total_steps=self._step_count,
                    timing_stats=self.action_timing.snapshot(),
                )
                self._on_task_complete(result)
                return result

            # 7. Inertia — check plan stability
            if self.inertia and self.inertia.has_plan:
                planned = self.inertia._commitment.next_planned_action
                if planned and planned != action_name:
                    inertia_decision = self.inertia.should_follow_plan(
                        planned, action_name,
                        new_action_confidence=routing_decision.confidence,
                    )
                    if inertia_decision.use_planned:
                        action_name = planned
                        logger.info(f"Inertia: keeping plan ({planned})")

            # 8. Build action
            action = self._build_action(action_name, params, screen)
            if action is None:
                steps.append(StepRecord(
                    step=self._step_count, screen_summary=screen_context[:200],
                    reasoning=reasoning, action=action_name, params=params,
                    success=False, error=f"Unknown or invalid action: {action_name}",
                ))
                self._success_window.append(False)
                continue

            # 9. Safety check
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
                    return TaskResult(
                        status=TaskStatus.BLOCKED, summary=result.message,
                        steps=steps, total_steps=self._step_count,
                        timing_stats=self.action_timing.snapshot(),
                    )

            # 10. Execute with retry + verify
            success, error, duration_ms, attempts = await self._run_action_with_retry(action)
            self.action_timing.record(action.type, duration_ms, success)
            self.action_recorder.record(
                timestamp_ms=int(time.time() * 1000), action=action,
                success=success, error=error, duration_ms=duration_ms, attempts=attempts,
            )
            self._success_window.append(success)

            steps.append(StepRecord(
                step=self._step_count, screen_summary=screen_context[:200],
                reasoning=reasoning, action=action_name, params=params,
                success=success, duration_ms=duration_ms, attempts=attempts, error=error,
            ))

            # 11. Explorer — update Q-values
            if self.explorer:
                reward = 1.0 if success else -0.5
                self.explorer.update_value(screen_hash, action_name, reward)
                self.explorer.anneal()

            # 12. Update knowledge graph
            if prev_screen_hash:
                new_screen = await self._observe()
                new_hash = ScreenGraph.compute_screen_hash(new_screen.raw_xml)
                self.graph.record_transition(prev_screen_hash, new_hash, action_name, success)
                if success and not used_cache:
                    self.router.cache_action(prev_screen_hash, action_json)
            else:
                new_hash = screen_hash

            prev_screen_hash = new_hash

            # 13. Advance inertia plan
            if self.inertia and success:
                self.inertia.advance()

            # 14. Circadian — record activity
            if self.circadian and screen.package:
                self.circadian.record_activity(screen.package, action_name, duration_ms=duration_ms)

            # 15. Homeostasis — update metrics
            if self.homeostasis:
                recent = self._success_window[-20:]
                sr = sum(recent) / len(recent) if recent else 0.5
                step_ms = (time.perf_counter() - step_start) * 1000
                err = 1.0 - sr
                self.homeostasis.update_metrics(
                    success_rate=sr, response_latency_ms=step_ms, error_rate=err,
                )

            # Periodic pheromone evaporation
            if self._step_count % 10 == 0:
                self.graph.evaporate_pheromones()

            # 16. Delay
            await asyncio.sleep(self.config.action_delay_ms / 1000)

        result = TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Max steps ({self.config.max_steps}) exceeded",
            steps=steps, total_steps=self._step_count,
            timing_stats=self.action_timing.snapshot(),
        )
        self._on_task_complete(result)
        return result

    def _on_task_complete(self, result: TaskResult) -> None:
        """Post-task hook: phase detector, graph persistence, genome encoding."""
        # Phase detector
        if self.phase_detector:
            recent = self._success_window[-20:]
            sr = sum(recent) / len(recent) if recent else 0.0
            snapshot = PerformanceSnapshot(
                timestamp=time.time(),
                success_rate=sr,
                avg_steps=result.total_steps,
                graph_nodes=len(self.graph._nodes),
                graph_edges=sum(len(e) for e in self.graph._edges.values()),
                task_count=1,
            )
            transition = self.phase_detector.record(snapshot)
            if transition:
                logger.info(
                    f"Phase transition: {transition.metric_name} "
                    f"{transition.old_value:.3f} -> {transition.new_value:.3f}"
                )

        # Persist knowledge graph
        if self.config.graph_persist_path:
            self.graph.save(self.config.graph_persist_path)

        # Encode completed task as workflow genome (for future evolution)
        if result.status == TaskStatus.COMPLETED and result.steps:
            genome = self._encode_task_genome(result)
            logger.debug(f"Task genome: {len(genome.codons)} codons, encoded={genome.encode()[:10]}...")

    def _encode_task_genome(self, result: TaskResult) -> Workflow:
        """Convert completed task steps into a workflow genome."""
        _ACTION_TO_OPCODE = {
            "tap": Opcode.TAP, "type_text": Opcode.TYP, "scroll": Opcode.SCR,
            "back": Opcode.BCK, "home": Opcode.HOM, "launch": Opcode.LCH,
            "wait": Opcode.WAT, "done": Opcode.DON,
        }
        codons = []
        text_table: dict[int, str] = {}
        text_idx = 0

        for step in result.steps:
            opcode = _ACTION_TO_OPCODE.get(step.action, Opcode.NOP)
            operand = 0
            if step.action == "tap":
                operand = step.params.get("index", 0)
            elif step.action == "type_text":
                text = step.params.get("text", "")
                text_table[text_idx] = text
                operand = text_idx
                text_idx += 1
            elif step.action == "launch":
                pkg = step.params.get("package", "")
                text_table[text_idx] = pkg
                operand = text_idx
                text_idx += 1
            elif step.action == "scroll":
                operand = 0 if step.params.get("direction") == "down" else 1
            codons.append(Codon(opcode, operand, step.reasoning[:30]))

        gene = Gene(name="task_trace", codons=codons)
        return Workflow(name=result.summary[:50], genes=[gene], text_table=text_table)

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
