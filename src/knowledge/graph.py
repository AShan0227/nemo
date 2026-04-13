"""Screen state graph — knowledge graph for app navigation.

Nodes = screen states (identified by activity + UI hash)
Edges = actions (tap, swipe, type, etc.) with cost weights
Used by A* planner to find optimal task paths.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScreenNode:
    """A known screen state in the navigation graph."""

    state_id: str
    activity: str = ""
    package: str = ""
    description: str = ""
    visit_count: int = 0
    ui_hash: str = ""


@dataclass
class ActionEdge:
    """An action that transitions between screen states."""

    from_state: str
    to_state: str
    action_type: str
    action_params: dict[str, Any] = field(default_factory=dict)
    cost: float = 1.0  # weighted cost (time + risk + compute)
    success_count: int = 0
    failure_count: int = 0
    pheromone: float = 1.0  # ACO pheromone for path reinforcement

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


class ScreenGraph:
    """Navigation knowledge graph with A* search."""

    def __init__(self) -> None:
        self._nodes: dict[str, ScreenNode] = {}
        self._edges: dict[str, list[ActionEdge]] = {}  # from_state -> [edges]

    def add_node(self, node: ScreenNode) -> None:
        self._nodes[node.state_id] = node

    def add_edge(self, edge: ActionEdge) -> None:
        if edge.from_state not in self._edges:
            self._edges[edge.from_state] = []
        self._edges[edge.from_state].append(edge)

    def get_neighbors(self, state_id: str) -> list[ActionEdge]:
        return self._edges.get(state_id, [])

    def record_transition(
        self, from_state: str, to_state: str, action_type: str, success: bool
    ) -> None:
        """Record a transition outcome, building the graph incrementally."""
        if from_state not in self._nodes:
            self.add_node(ScreenNode(state_id=from_state))
        if to_state not in self._nodes:
            self.add_node(ScreenNode(state_id=to_state))

        # Find or create edge
        edges = self.get_neighbors(from_state)
        matching = [e for e in edges if e.to_state == to_state and e.action_type == action_type]

        if matching:
            edge = matching[0]
        else:
            edge = ActionEdge(from_state=from_state, to_state=to_state, action_type=action_type)
            self.add_edge(edge)

        if success:
            edge.success_count += 1
            edge.pheromone = min(edge.pheromone * 1.1, 10.0)
        else:
            edge.failure_count += 1
            edge.pheromone = max(edge.pheromone * 0.9, 0.1)

        # Update edge cost based on success rate
        edge.cost = 1.0 / max(edge.success_rate, 0.01)

    def a_star(self, start: str, goal: str, heuristic_fn=None) -> list[ActionEdge] | None:
        """Find optimal path from start to goal screen state."""
        if start == goal:
            return []

        if heuristic_fn is None:
            heuristic_fn = lambda s, g: 0  # Dijkstra fallback

        open_set: list[tuple[float, str, list[ActionEdge]]] = [(0.0, start, [])]
        visited: set[str] = set()

        while open_set:
            cost, current, path = heapq.heappop(open_set)

            if current == goal:
                return path

            if current in visited:
                continue
            visited.add(current)

            for edge in self.get_neighbors(current):
                if edge.to_state not in visited:
                    new_cost = cost + edge.cost
                    h = heuristic_fn(edge.to_state, goal)
                    heapq.heappush(open_set, (new_cost + h, edge.to_state, path + [edge]))

        return None  # no path found

    def evaporate_pheromones(self, rate: float = 0.05) -> None:
        """ACO pheromone evaporation — prevents lock-in to suboptimal paths."""
        for edges in self._edges.values():
            for edge in edges:
                edge.pheromone *= 1.0 - rate

    @staticmethod
    def compute_screen_hash(xml_dump: str) -> str:
        """Hash a screen state for deduplication."""
        return hashlib.md5(xml_dump.encode()).hexdigest()[:12]
