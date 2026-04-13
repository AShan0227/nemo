"""Thermodynamic Explorer — entropy-regularized action selection (SAC-inspired).

Balances exploration vs exploitation using a temperature-scaled
Boltzmann distribution over action values. Higher temperature =
more exploration, lower = more exploitation.

Inspired by Soft Actor-Critic (SAC):
  J(π) = E[Σ r(s,a) + α * H(π(·|s))]

The entropy bonus α * H(π) encourages visiting unknown states.
Temperature α anneals over time as the agent learns.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ActionValue:
    """Value estimate for an action in a given state."""
    action_name: str
    q_value: float = 0.0
    visit_count: int = 0


class BoltzmannExplorer:
    """Temperature-scaled Boltzmann action selection.

    Actions are selected with probability proportional to:
        P(a|s) ∝ exp(Q(s,a) / τ)

    where τ is the temperature parameter. τ → 0 = greedy, τ → ∞ = uniform.
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        annealing_rate: float = 0.995,
        entropy_bonus: float = 0.1,
    ) -> None:
        self._temperature = initial_temperature
        self._min_temp = min_temperature
        self._annealing_rate = annealing_rate
        self._entropy_bonus = entropy_bonus
        self._q_table: Dict[str, List[ActionValue]] = {}  # state -> action values
        self._total_steps = 0

    @property
    def temperature(self) -> float:
        return self._temperature

    def get_action_values(self, state_hash: str) -> List[ActionValue]:
        """Get known action values for a state."""
        return self._q_table.get(state_hash, [])

    def select_action(
        self, state_hash: str, available_actions: List[str]
    ) -> Tuple[str, float]:
        """Select action using Boltzmann distribution.

        Returns (action_name, selection_probability).
        """
        if not available_actions:
            return ("wait", 1.0)

        values = self._q_table.get(state_hash, [])
        value_map = {v.action_name: v.q_value for v in values}

        # Compute Boltzmann probabilities
        logits = []
        for action in available_actions:
            q = value_map.get(action, 0.0)
            # Add entropy bonus for unvisited actions
            visits = next((v.visit_count for v in values if v.action_name == action), 0)
            exploration_bonus = self._entropy_bonus / max(visits, 1) ** 0.5
            logits.append((q + exploration_bonus) / max(self._temperature, 0.01))

        # Stable softmax
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        # Sample
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if cumulative >= r:
                self._total_steps += 1
                return (available_actions[i], probs[i])

        return (available_actions[-1], probs[-1])

    def update_value(
        self,
        state_hash: str,
        action_name: str,
        reward: float,
        learning_rate: float = 0.1,
    ) -> None:
        """Update Q-value for (state, action) pair."""
        if state_hash not in self._q_table:
            self._q_table[state_hash] = []

        values = self._q_table[state_hash]
        existing = next((v for v in values if v.action_name == action_name), None)

        if existing:
            # Incremental Q-update
            existing.q_value += learning_rate * (reward - existing.q_value)
            existing.visit_count += 1
        else:
            values.append(ActionValue(
                action_name=action_name,
                q_value=reward,
                visit_count=1,
            ))

    def anneal(self) -> None:
        """Reduce temperature (less exploration over time)."""
        self._temperature = max(
            self._min_temp,
            self._temperature * self._annealing_rate,
        )

    def compute_policy_entropy(self, state_hash: str, actions: List[str]) -> float:
        """Compute Shannon entropy of the current policy for a state."""
        if not actions:
            return 0.0

        values = self._q_table.get(state_hash, [])
        value_map = {v.action_name: v.q_value for v in values}

        logits = [value_map.get(a, 0.0) / max(self._temperature, 0.01) for a in actions]
        max_logit = max(logits)
        exp_logits = [math.exp(l - max_logit) for l in logits]
        total = sum(exp_logits)
        probs = [e / total for e in exp_logits]

        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @property
    def stats(self) -> Dict[str, float]:
        return {
            "temperature": self._temperature,
            "total_steps": self._total_steps,
            "known_states": len(self._q_table),
            "total_actions": sum(len(v) for v in self._q_table.values()),
        }
