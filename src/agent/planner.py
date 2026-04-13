"""Task planner — A* path planning over screen state graph.

Based on Principle of Least Action (Lagrangian mechanics):
- cost(state, action) = w1*time + w2*risk + w3*compute + w4*(1-reversibility)
- Uses A* with learned heuristic for optimal task paths
- Falls back to LLM planning for unknown territories
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.knowledge.graph import ScreenGraph
from src.llm.prompts import PLANNING_PROMPT


@dataclass
class PlanStep:
    """A single step in a task plan."""

    step_number: int
    action_type: str
    params: dict[str, Any]
    description: str
    expected_state: str
    estimated_cost: float


@dataclass
class TaskPlan:
    """Complete plan for a task."""

    task: str
    steps: list[PlanStep]
    total_cost: float
    confidence: float  # 0-1, based on graph coverage


class TaskPlanner:
    """Plan optimal task execution using knowledge graph."""

    def __init__(self, graph: ScreenGraph) -> None:
        self._graph = graph

    def plan_from_graph(self, current_state: str, goal_state: str) -> TaskPlan | None:
        """Try to find a plan using A* on the knowledge graph."""
        path = self._graph.a_star(current_state, goal_state)
        if path is None:
            return None

        steps = []
        total_cost = 0.0
        for i, edge in enumerate(path):
            steps.append(PlanStep(
                step_number=i + 1,
                action_type=edge.action_type,
                params={},
                description=f"{edge.action_type} -> {edge.to_state}",
                expected_state=edge.to_state,
                estimated_cost=edge.cost,
            ))
            total_cost += edge.cost

        # Confidence based on success rates of edges in the path
        avg_success = sum(e.success_rate for e in path) / len(path) if path else 1.0

        return TaskPlan(
            task=f"{current_state} -> {goal_state}",
            steps=steps,
            total_cost=total_cost,
            confidence=avg_success,
        )

    async def plan_with_llm(
        self,
        llm: Any,
        *,
        task: str,
        current_app: str,
        screen_summary: str,
    ) -> TaskPlan | None:
        """Fallback: ask LLM for coarse-grained multi-step plan."""
        prompt = PLANNING_PROMPT.format(
            task=task,
            current_app=current_app or "unknown",
            screen_summary=screen_summary,
        )
        response = await llm.chat(
            messages=[
                {"role": "system", "content": "You produce concise executable plans."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.get("content", "")
        parsed = self._extract_json_array(raw)
        if parsed is None:
            return None

        steps: list[PlanStep] = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            action = str(item.get("action_name", item.get("action", ""))).strip()
            if not action:
                continue
            params = item.get("params", {})
            if not isinstance(params, dict):
                params = {}
            steps.append(
                PlanStep(
                    step_number=i + 1,
                    action_type=action,
                    params=params,
                    description=str(item.get("action", action)),
                    expected_state=str(item.get("expected_result", "")),
                    estimated_cost=float(item.get("estimated_cost", 1.0)),
                )
            )

        if not steps:
            return None

        return TaskPlan(
            task=task,
            steps=steps,
            total_cost=sum(step.estimated_cost for step in steps),
            confidence=0.5,
        )

    @staticmethod
    def next_action_hint(plan: TaskPlan | None, cursor: int) -> PlanStep | None:
        if plan is None:
            return None
        if cursor < 0 or cursor >= len(plan.steps):
            return None
        return plan.steps[cursor]

    @staticmethod
    def _extract_json_array(raw: str) -> list[Any] | None:
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, list) else None
