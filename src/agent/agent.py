"""PhoneAgent — the main agent loop.

Observe → Think → Act → Verify → Learn cycle with research mechanisms:
1. Capture screen state (UI hierarchy + OCR + visual fusion)
2. Immune check: detect anomalous screens
3. Homeostasis + Circadian: adjust behavior based on metrics and time
4. Route to reasoning depth (entropy router)
5. Planner: LLM fallback planning for unknown territories
6. LLM decides next action (structured tool_use or legacy JSON)
7. Inertia: check plan stability before accepting deviation
8. Safety layer validates action
9. Execute action (with retry + screenshot verify)
10. Explorer: update Boltzmann Q-values
11. Update knowledge graph + cache + genome encode
12. Phase detector: check for capability jumps
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
from src.llm.prompts import resolve_system_prompt
from src.llm.router import EntropyRouter, ReasoningDepth, RoutingDecision
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
from src.screen.parser import ScreenParserCache, ScreenState


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class StepRecord:
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
        self.immune = (
            ImmuneSystem(anomaly_threshold=config.immune_anomaly_threshold)
            if config.immune_enabled
            else None
        )
        self.inertia = (
            InertiaController(base_inertia=config.inertia_base)
            if config.inertia_enabled
            else None
        )
        self.explorer = (
            BoltzmannExplorer(initial_temperature=config.explorer_temperature)
            if config.explorer_enabled
            else None
        )
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
        self._load_graph_from_disk()
        logger.info(f"PhoneAgent ready (screen: {self._screen_size[0]}x{self._screen_size[1]})")

    async def replay_recording(
        self, path: str | Path, *, speed: float = 1.0,
        preserve_timing: bool = True, stop_on_error: bool = True,
    ) -> ReplayReport:
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
        active_plan = None
        plan_cursor = 0
        self.action_recorder.clear()
        self.action_timing = ActionTimingTracker()

        while self._step_count < self.config.max_steps:
            self._step_count += 1
            logger.info(f"--- Step {self._step_count} ---")
            step_start = time.perf_counter()

            # 1. Observe
            screen = await self._observe()
            screen_hash = ScreenGraph.compute_screen_hash(screen.raw_xml)

            if screen_hash not in self.graph._nodes:
                self.graph.add_node(ScreenNode(
                    state_id=screen_hash, activity=screen.activity,
                    package=screen.package, ui_hash=screen_hash,
                ))
            self.graph._nodes[screen_hash].visit_count += 1

            # 2. Immune check
            if self.immune and self.immune._trained:
                features = extract_features(screen)
                anomaly = self.immune.check(features)
                if anomaly.is_anomaly:
                    logger.warning(f"Anomaly detected: {anomaly.message}. Pressing back.")
                    await self.adb.press_back()
                    await asyncio.sleep(0.5)
                    continue
            elif self.immune and not self.immune._trained:
                self.immune.add_self_sample(extract_features(screen))
                if len(self.immune._self_set) >= 20:
                    self.immune.train()

            # 3. Homeostasis + Circadian
            if self.homeostasis:
                for adj in self.homeostasis.get_adjustments():
                    if adj.name == "increase_delay" and adj.value > 0.1:
                        self.config.action_delay_ms = min(2000, self.config.action_delay_ms + 100)
            if self.circadian:
                mods = self.circadian.get_behavior_modifier()
                self.config.action_delay_ms = min(
                    3000, int(self.config.action_delay_ms * mods["action_delay_multiplier"]))

            # 4. Route
            screen_context = await self._build_screen_context(screen)
            if self.config.router_enabled:
                routing_decision = self._route_decision(screen_hash)
            else:
                routing_decision = RoutingDecision(
                    depth=ReasoningDepth.SYSTEM_2,
                    entropy=1.0,
                    confidence=0.0,
                    source="router_disabled",
                )

            # 4b. Planner LLM fallback (first step only)
            if self.config.planner_enabled and active_plan is None:
                planner_chat = self._resolve_llm_method("chat")
                if planner_chat is not None:
                    try:
                        active_plan = await self.planner.plan_with_llm(
                            self.llm, task=task,
                            current_app=screen.activity, screen_summary=screen_context,
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

            # 5. Think — planner hint, structured tool_use, or legacy JSON
            if action_json is None and decision is None:
                history_payload = self._build_recent_context(steps)
                prompt_version = getattr(self.config.llm, "prompt_version", "reflect_fewshot_v1")
                system_prompt = resolve_system_prompt(prompt_version)
                planned_hint = self.planner.next_action_hint(active_plan, plan_cursor)
                if (self.config.planner_enabled and planned_hint is not None
                        and routing_decision.depth != ReasoningDepth.SYSTEM_2):
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
                    # Try structured tool_use first
                    structured_method = self._resolve_llm_method("decide_action_structured")
                    if structured_method is not None:
                        try:
                            payload = await self._call_llm_method(
                                structured_method,
                                system_prompt,
                                screen_context,
                                task,
                                history=history_payload,
                                prompt_version=prompt_version,
                            )
                            if isinstance(payload, dict):
                                decision = payload
                            elif isinstance(payload, str):
                                action_json = payload
                        except Exception:
                            pass

                    # Fallback to legacy decide_action
                    if decision is None and action_json is None:
                        decide_method = self._resolve_llm_method("decide_action")
                        if decide_method is None:
                            raise RuntimeError("LLM client missing `decide_action` interface")
                        legacy = await self._call_llm_method(
                            decide_method,
                            system_prompt,
                            screen_context,
                            task,
                            history=history_payload,
                            prompt_version=prompt_version,
                        )
                        if isinstance(legacy, dict):
                            decision = legacy
                        else:
                            action_json = str(legacy)

            # 6. Parse
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
                    self._success_window.append(False)
                    continue

            reasoning = decision.get("reasoning", "")
            action_name = decision.get("action", "")
            params = decision.get("params", {})
            meta = decision.get("_meta", {}) if isinstance(decision, dict) else {}
            if isinstance(meta, dict):
                entropy = meta.get("entropy")
                if isinstance(entropy, (int, float)):
                    self.router.observe_entropy(screen_hash, float(entropy))

            # Done check
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

            # 7. Inertia
            if self.inertia and self.inertia.has_plan:
                planned = self.inertia._commitment.next_planned_action
                if planned and planned != action_name:
                    inertia_decision = self.inertia.should_follow_plan(
                        planned, action_name, new_action_confidence=routing_decision.confidence)
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
                    blocked_result = TaskResult(
                        status=TaskStatus.BLOCKED, summary=result.message,
                        steps=steps, total_steps=self._step_count,
                        timing_stats=self.action_timing.snapshot(),
                    )
                    self._save_graph_to_disk()
                    return blocked_result

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

            # 11. Explorer Q-values
            if self.explorer:
                self.explorer.update_value(screen_hash, action_name, 1.0 if success else -0.5)
                self.explorer.anneal()

            # 12. Update knowledge graph
            if prev_screen_hash:
                new_screen = await self._observe()
                new_hash = ScreenGraph.compute_screen_hash(new_screen.raw_xml)
                self.graph.record_transition(prev_screen_hash, new_hash, action_name, success)
                if success and not used_cache:
                    self.router.cache_action(prev_screen_hash, action_json or json.dumps(decision))
            else:
                new_hash = screen_hash
            prev_screen_hash = new_hash

            # 13. Inertia advance + circadian record
            if self.inertia and success:
                self.inertia.advance()
            if self.circadian and screen.package:
                self.circadian.record_activity(screen.package, action_name, duration_ms=duration_ms)

            # 14. Homeostasis metrics
            if self.homeostasis:
                recent = self._success_window[-20:]
                sr = sum(recent) / len(recent) if recent else 0.5
                step_ms = (time.perf_counter() - step_start) * 1000
                self.homeostasis.update_metrics(
                    success_rate=sr,
                    response_latency_ms=step_ms,
                    error_rate=1.0 - sr,
                )

            if self._step_count % 10 == 0:
                self.graph.evaporate_pheromones()

            await asyncio.sleep(self.config.action_delay_ms / 1000)

        failed_result = TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Max steps ({self.config.max_steps}) exceeded",
            steps=steps, total_steps=self._step_count,
            timing_stats=self.action_timing.snapshot(),
        )
        self._on_task_complete(failed_result)
        return failed_result

    # --- Post-task hooks ---

    def _on_task_complete(self, result: TaskResult) -> None:
        if self.phase_detector:
            recent = self._success_window[-20:]
            sr = sum(recent) / len(recent) if recent else 0.0
            transition = self.phase_detector.record(PerformanceSnapshot(
                timestamp=time.time(), success_rate=sr, avg_steps=result.total_steps,
                graph_nodes=len(self.graph._nodes),
                graph_edges=sum(len(e) for e in self.graph._edges.values()), task_count=1,
            ))
            if transition:
                logger.info(f"Phase transition: {transition.metric_name} "
                            f"{transition.old_value:.3f} -> {transition.new_value:.3f}")
        self._save_graph_to_disk()
        if result.status == TaskStatus.COMPLETED and result.steps:
            genome = self._encode_task_genome(result)
            logger.debug(f"Task genome: {len(genome.codons)} codons")

    def _encode_task_genome(self, result: TaskResult) -> Workflow:
        _ACTION_TO_OPCODE = {
            "tap": Opcode.TAP, "type_text": Opcode.TYP, "scroll": Opcode.SCR,
            "back": Opcode.BCK, "home": Opcode.HOM, "launch": Opcode.LCH,
            "wait": Opcode.WAT, "done": Opcode.DON,
        }
        codons, text_table, text_idx = [], {}, 0
        for step in result.steps:
            opcode = _ACTION_TO_OPCODE.get(step.action, Opcode.NOP)
            operand = 0
            if step.action == "tap":
                operand = step.params.get("index", 0)
            elif step.action in ("type_text", "launch"):
                key = "text" if step.action == "type_text" else "package"
                text_table[text_idx] = step.params.get(key, "")
                operand = text_idx
                text_idx += 1
            elif step.action == "scroll":
                operand = 0 if step.params.get("direction") == "down" else 1
            codons.append(Codon(opcode, operand, step.reasoning[:30]))
        return Workflow(
            name=result.summary[:50],
            genes=[Gene("task_trace", codons)],
            text_table=text_table,
        )

    # --- Graph persistence ---

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
        try:
            self.graph.save(path_str)
        except Exception as exc:
            logger.warning("Failed to save graph to {}: {}", path_str, exc)

    # --- LLM method resolution (supports structured tool_use + legacy) ---

    def _resolve_llm_method(self, name: str) -> Callable[..., Any] | None:
        instance_dict = getattr(self.llm, "__dict__", {})
        has_instance_attr = isinstance(instance_dict, dict) and name in instance_dict
        has_class_attr = hasattr(type(self.llm), name)
        if not (has_instance_attr or has_class_attr):
            return None
        method = getattr(self.llm, name, None)
        return method if callable(method) else None

    async def _call_llm_method(self, method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call LLM method with kwargs when supported; keep compatibility with old mocks."""
        try:
            return await method(*args, **kwargs)
        except TypeError as exc:
            text = str(exc)
            keyword_mismatch = (
                "unexpected keyword" in text
                or "positional arguments but" in text
                or "got an unexpected keyword argument" in text
            )
            if kwargs and keyword_mismatch:
                return await method(*args)
            raise

    # --- Action execution ---

    async def _run_action_with_retry(self, action: Action) -> tuple[bool, str, float, int]:
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
                        before_path, after_path, threshold=self.config.verify_diff_threshold)
                    if not changed:
                        raise RuntimeError("verify_failed: screen appears unchanged")
                return True, "", (time.perf_counter() - started) * 1000, attempt
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Action {} failed attempt {}/{}: {}",
                    action.type.value,
                    attempt,
                    max_attempts,
                    last_error,
                )
                if attempt >= max_attempts:
                    break
                await asyncio.sleep(0.2)
        return False, last_error, (time.perf_counter() - started) * 1000, max_attempts

    def _should_verify(self, action: Action) -> bool:
        return self.config.verify_enabled and action.type != ActionType.WAIT

    async def _observe(self) -> ScreenState:
        xml = await self.adb.get_ui_hierarchy()
        state, dedup_hit = self.screen_parser.parse(xml)
        if dedup_hit:
            logger.debug("Screen hash unchanged, reusing last parsed state")
        return state

    async def _build_screen_context(self, screen: ScreenState) -> str:
        context = screen.to_prompt_str()
        if not self.config.ocr_enabled:
            return context
        try:
            screenshot = await self.adb.screenshot()
            ocr_blocks = await asyncio.to_thread(self.ocr.extract, screenshot)
        except Exception as exc:
            logger.warning(f"Failed to run OCR: {exc}")
            return context
        if not ocr_blocks:
            return context
        fusion = fuse_screen_sources(
            ui_candidates=candidates_from_ui(screen.interactive_elements),
            ocr_candidates=candidates_from_ocr(
                ocr_blocks,
                min_confidence=self.config.ocr_min_confidence,
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
        return self.router.route(screen_hash)

    def _build_recent_context(self, steps: list[StepRecord]) -> list[dict[str, Any]]:
        """Build recent trajectory context for multi-turn LLM grounding."""
        window = max(0, int(getattr(self.config, "context_window_steps", 0)))
        if window == 0 or not steps:
            return []

        recent = steps[-window:]
        context: list[dict[str, Any]] = []
        for item in recent:
            context.append(
                {
                    "step": item.step,
                    "action": item.action,
                    "success": item.success,
                    "error": item.error,
                    "params": item.params,
                }
            )
        return context

    def _parse_decision(self, raw: str) -> dict[str, Any]:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise ValueError(f"No JSON found in response: {raw[:200]}")

    def _build_action(self, name: str, params: dict, screen: ScreenState) -> Action | None:
        elems = screen.interactive_elements

        def _resolve_center(index: int) -> tuple[int, int] | None:
            return elems[index].center if 0 <= index < len(elems) else None

        if name == "tap":
            center = _resolve_center(params.get("index", 0))
            if center:
                return Action.tap(center[0], center[1], f"Tap [{params.get('index', 0)}]")
        elif name == "type_text":
            idx, text = params.get("index", 0), params.get("text", "")
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
        t = action.type
        if t == ActionType.TAP:
            await self.adb.tap(action.params["x"], action.params["y"])
        elif t == ActionType.SWIPE:
            await self.adb.swipe(action.params["x1"], action.params["y1"],
                                 action.params["x2"], action.params["y2"],
                                 action.params.get("duration_ms", 300))
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
