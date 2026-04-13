"""Task planner — A* path planning over screen state graph.

Based on Principle of Least Action (Lagrangian mechanics):
- cost(state, action) = w1*time + w2*risk + w3*compute + w4*(1-reversibility)
- Uses A* with learned heuristic for optimal task paths
- Falls back to LLM planning for unknown territories
"""

from __future__ import annotations

from dataclasses import dataclass

from src.knowledge.graph import ActionEdge, ScreenGraph


@dataclass
class PlanStep:
    """A single step in a task plan."""

    step_number: int
    action_type: str
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
                description=f"{edge.action_type} -> {edge.to_state}",
                expected_state=edge.to_state,
                estimated_cost=edge.cost,
            ))
            total_cost += edge.cost

        # Confidence based on success rates of edges in the path
        if path:
            avg_success = sum(e.success_rate for e in path) / len(path)
        else:
            avg_success = 1.0

        return TaskPlan(
            task=f"{current_state} -> {goal_state}",
            steps=steps,
            total_cost=total_cost,
            confidence=avg_success,
        )
