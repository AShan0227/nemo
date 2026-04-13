"""Screen state graph — knowledge graph for app navigation.

Nodes = screen states (identified by activity + UI hash)
Edges = actions (tap, swipe, type, etc.) with cost weights
Uses A* for optimal path planning and ACO for probabilistic action selection.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
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
            def _zero_heuristic(s: str, g: str) -> float:
                return 0.0

            heuristic_fn = _zero_heuristic  # Dijkstra fallback

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

    def select_action_aco(
        self,
        state_id: str,
        alpha: float = 1.0,
        beta: float = 2.0,
        q0: float = 0.9,
    ) -> ActionEdge | None:
        """ACO action selection with Ant Colony System pseudo-random proportional rule.

        With probability q0, greedily pick the best edge (exploitation).
        With probability 1-q0, sample proportionally (exploration).

        Args:
            state_id: Current screen state hash
            alpha: Pheromone importance weight
            beta: Heuristic desirability weight (1/cost)
            q0: Exploitation probability (0-1). Higher = more greedy.
        """
        edges = self.get_neighbors(state_id)
        if not edges:
            return None
        if len(edges) == 1:
            return edges[0]

        # Compute attractiveness for each edge
        scores = []
        for e in edges:
            tau = e.pheromone ** alpha
            eta = (1.0 / max(e.cost, 0.01)) ** beta
            scores.append(tau * eta)

        # ACS: pseudo-random proportional rule
        if random.random() < q0:
            # Exploitation: pick the edge with highest score
            best_idx = max(range(len(edges)), key=lambda i: scores[i])
            return edges[best_idx]
        else:
            # Exploration: roulette wheel selection
            total = sum(scores)
            if total == 0:
                return random.choice(edges)
            pick = random.random() * total
            cumulative = 0.0
            for i, s in enumerate(scores):
                cumulative += s
                if cumulative >= pick:
                    return edges[i]
            return edges[-1]

    def evaporate_pheromones(self, rate: float = 0.05) -> None:
        """ACO pheromone evaporation — prevents lock-in to suboptimal paths."""
        for edges in self._edges.values():
            for edge in edges:
                edge.pheromone = max(edge.pheromone * (1.0 - rate), 0.1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "state_id": node.state_id,
                    "activity": node.activity,
                    "package": node.package,
                    "description": node.description,
                    "visit_count": node.visit_count,
                    "ui_hash": node.ui_hash,
                }
                for node in self._nodes.values()
            ],
            "edges": [
                {
                    "from_state": edge.from_state,
                    "to_state": edge.to_state,
                    "action_type": edge.action_type,
                    "action_params": edge.action_params,
                    "cost": edge.cost,
                    "success_count": edge.success_count,
                    "failure_count": edge.failure_count,
                    "pheromone": edge.pheromone,
                }
                for edges in self._edges.values()
                for edge in edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreenGraph:
        graph = cls()
        for node_data in data.get("nodes", []):
            if not isinstance(node_data, dict):
                continue
            node = ScreenNode(
                state_id=str(node_data.get("state_id", "")),
                activity=str(node_data.get("activity", "")),
                package=str(node_data.get("package", "")),
                description=str(node_data.get("description", "")),
                visit_count=int(node_data.get("visit_count", 0)),
                ui_hash=str(node_data.get("ui_hash", "")),
            )
            if node.state_id:
                graph.add_node(node)

        for edge_data in data.get("edges", []):
            if not isinstance(edge_data, dict):
                continue
            edge = ActionEdge(
                from_state=str(edge_data.get("from_state", "")),
                to_state=str(edge_data.get("to_state", "")),
                action_type=str(edge_data.get("action_type", "")),
                action_params=dict(edge_data.get("action_params", {})),
                cost=float(edge_data.get("cost", 1.0)),
                success_count=int(edge_data.get("success_count", 0)),
                failure_count=int(edge_data.get("failure_count", 0)),
                pheromone=float(edge_data.get("pheromone", 1.0)),
            )
            if edge.from_state and edge.to_state and edge.action_type:
                graph.add_edge(edge)
        return graph

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> ScreenGraph:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Graph file must contain a top-level JSON object")
        return cls.from_dict(payload)

    @staticmethod
    def compute_screen_hash(xml_dump: str) -> str:
        """Hash a screen state for deduplication."""
        return hashlib.md5(xml_dump.encode()).hexdigest()[:12]
