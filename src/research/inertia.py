"""Friction & Inertia — plan stability for agent execution.

Prevents the agent from oscillating between different plans mid-execution.
Once a plan is committed, applies an inertia penalty for switching plans,
analogous to physical inertia resisting changes in motion.

Key idea: new_action must be significantly better than planned_action
to justify a plan deviation. The threshold adapts based on how far
along the plan the agent is (more inertia = further into plan).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class PlanCommitment:
    """Tracks the agent's commitment to a plan."""
    planned_actions: List[str]
    current_step: int = 0
    deviations: int = 0
    total_steps_executed: int = 0

    @property
    def progress(self) -> float:
        """How far along the plan (0-1)."""
        if not self.planned_actions:
            return 0.0
        return min(self.current_step / len(self.planned_actions), 1.0)

    @property
    def deviation_rate(self) -> float:
        if self.total_steps_executed == 0:
            return 0.0
        return self.deviations / self.total_steps_executed

    @property
    def next_planned_action(self) -> Optional[str]:
        if self.current_step < len(self.planned_actions):
            return self.planned_actions[self.current_step]
        return None


@dataclass
class InertiaDecision:
    """Result of inertia check."""
    use_planned: bool
    reason: str
    inertia_score: float  # 0-1, how much inertia was applied
    override_cost: float  # how much better the new action needs to be


class InertiaController:
    """Apply inertia to prevent excessive plan switching.

    Inertia increases as:
    - Plan progress increases (don't abandon a nearly-complete plan)
    - Plan has been successful so far (low deviation rate)
    - Base inertia parameter is higher

    Inertia decreases when:
    - The current step has failed multiple times
    - The plan deviation rate is already high
    """

    def __init__(
        self,
        base_inertia: float = 0.3,
        progress_weight: float = 0.5,
        max_inertia: float = 0.8,
    ) -> None:
        self._base_inertia = base_inertia
        self._progress_weight = progress_weight
        self._max_inertia = max_inertia
        self._commitment: Optional[PlanCommitment] = None

    def set_plan(self, actions: List[str]) -> None:
        """Commit to a new plan."""
        self._commitment = PlanCommitment(planned_actions=actions)
        logger.debug(f"Plan committed: {len(actions)} steps")

    def clear_plan(self) -> None:
        """Clear current plan (task completed or abandoned)."""
        self._commitment = None

    @property
    def has_plan(self) -> bool:
        return self._commitment is not None and self._commitment.next_planned_action is not None

    def compute_inertia(self) -> float:
        """Compute current inertia score (0 = no resistance, 1 = max resistance)."""
        if self._commitment is None:
            return 0.0

        # Base inertia + progress-scaled component
        progress_bonus = self._commitment.progress * self._progress_weight

        # Reduce inertia if plan is failing
        deviation_penalty = self._commitment.deviation_rate * 0.5

        inertia = self._base_inertia + progress_bonus - deviation_penalty
        return max(0.0, min(inertia, self._max_inertia))

    def should_follow_plan(
        self,
        planned_action: str,
        new_action: str,
        new_action_confidence: float = 0.5,
    ) -> InertiaDecision:
        """Decide whether to follow the plan or accept a new action.

        Args:
            planned_action: What the plan says to do next
            new_action: What the LLM suggests doing instead
            new_action_confidence: LLM's confidence in the new action (0-1)
        """
        if planned_action == new_action:
            # Same action — no conflict
            return InertiaDecision(
                use_planned=True,
                reason="LLM agrees with plan",
                inertia_score=0.0,
                override_cost=0.0,
            )

        inertia = self.compute_inertia()
        override_threshold = inertia

        if new_action_confidence > override_threshold:
            # New action is confident enough to overcome inertia
            if self._commitment:
                self._commitment.deviations += 1
            logger.debug(
                f"Plan override: {planned_action} -> {new_action} "
                f"(confidence={new_action_confidence:.2f} > inertia={inertia:.2f})"
            )
            return InertiaDecision(
                use_planned=False,
                reason=f"LLM override (confidence {new_action_confidence:.2f} > inertia {inertia:.2f})",
                inertia_score=inertia,
                override_cost=new_action_confidence - inertia,
            )
        else:
            logger.debug(
                f"Inertia holds: keeping {planned_action} "
                f"(confidence={new_action_confidence:.2f} < inertia={inertia:.2f})"
            )
            return InertiaDecision(
                use_planned=True,
                reason=f"Inertia holds (confidence {new_action_confidence:.2f} < inertia {inertia:.2f})",
                inertia_score=inertia,
                override_cost=0.0,
            )

    def advance(self) -> None:
        """Advance plan to next step after successful execution."""
        if self._commitment:
            self._commitment.current_step += 1
            self._commitment.total_steps_executed += 1
