"""Genetic Algorithm — strategy evolution for agent behavior.

Evolves task execution strategies by treating agent parameters as a genome.
Each individual is a parameter vector that controls agent behavior:
- action_delay, scroll_speed, verification_frequency, retry_count, etc.

Uses tournament selection, crossover, and mutation to evolve better strategies
over multiple task executions.

Based on: De Castro & Von Zuben, 2002 — evolutionary optimization.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class StrategyGenome:
    """A single execution strategy encoded as a parameter vector."""

    genes: Dict[str, float]
    fitness: float = 0.0
    task_count: int = 0
    generation: int = 0

    @staticmethod
    def default() -> StrategyGenome:
        """Create a genome with default parameter values."""
        return StrategyGenome(genes={
            "action_delay_ms": 500.0,
            "scroll_amount": 0.25,        # fraction of screen height
            "max_retries": 2.0,
            "verify_after_action": 1.0,    # 1.0 = always, 0.0 = never
            "exploration_rate": 0.1,       # ACO q0 inverse
            "prompt_detail_level": 1.0,    # 1.0 = full, 0.5 = condensed
            "safety_strictness": 1.0,      # 1.0 = strict, 0.5 = relaxed
            "focus_tap_delay_ms": 300.0,   # delay between tap-to-focus and type
        })

    @staticmethod
    def random() -> StrategyGenome:
        """Create a genome with randomized parameters."""
        base = StrategyGenome.default()
        for key in base.genes:
            base.genes[key] *= random.uniform(0.3, 2.0)
        return base

    def mutate(self, rate: float = 0.1, strength: float = 0.2) -> None:
        """Apply gaussian mutation to each gene with given probability."""
        for key in self.genes:
            if random.random() < rate:
                delta = self.genes[key] * strength * random.gauss(0, 1)
                self.genes[key] = max(0.01, self.genes[key] + delta)

    def get(self, key: str, default: float = 0.0) -> float:
        return self.genes.get(key, default)


def crossover_uniform(
    parent1: StrategyGenome, parent2: StrategyGenome
) -> Tuple[StrategyGenome, StrategyGenome]:
    """Uniform crossover — each gene randomly picked from either parent."""
    child1_genes = {}
    child2_genes = {}

    for key in parent1.genes:
        if random.random() < 0.5:
            child1_genes[key] = parent1.genes[key]
            child2_genes[key] = parent2.genes[key]
        else:
            child1_genes[key] = parent2.genes[key]
            child2_genes[key] = parent1.genes[key]

    return (
        StrategyGenome(genes=child1_genes),
        StrategyGenome(genes=child2_genes),
    )


def crossover_blend(
    parent1: StrategyGenome, parent2: StrategyGenome, alpha: float = 0.5
) -> StrategyGenome:
    """BLX-alpha crossover — child genes are blended from parents."""
    child_genes = {}
    for key in parent1.genes:
        v1, v2 = parent1.genes[key], parent2.genes[key]
        low = min(v1, v2) - alpha * abs(v2 - v1)
        high = max(v1, v2) + alpha * abs(v2 - v1)
        child_genes[key] = max(0.01, random.uniform(low, high))
    return StrategyGenome(genes=child_genes)


class StrategyEvolver:
    """Evolve agent execution strategies via genetic algorithm.

    Usage:
        evolver = StrategyEvolver(population_size=20)
        strategy = evolver.get_strategy()
        # ... run task with strategy parameters ...
        evolver.report_fitness(strategy, fitness_score)
        evolver.evolve()  # create next generation
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.15,
        mutation_strength: float = 0.2,
        elitism_count: int = 2,
        tournament_size: int = 3,
    ) -> None:
        self._pop_size = population_size
        self._mutation_rate = mutation_rate
        self._mutation_strength = mutation_strength
        self._elitism = elitism_count
        self._tournament_size = tournament_size
        self._generation = 0

        # Initialize population
        self._population: List[StrategyGenome] = [
            StrategyGenome.default()  # keep one default
        ]
        for _ in range(population_size - 1):
            self._population.append(StrategyGenome.random())

    def get_strategy(self) -> StrategyGenome:
        """Get the best current strategy for task execution."""
        evaluated = [g for g in self._population if g.task_count > 0]
        if evaluated:
            return max(evaluated, key=lambda g: g.fitness)
        # No evaluations yet — return a random one
        return random.choice(self._population)

    def report_fitness(
        self,
        genome: StrategyGenome,
        success: bool,
        steps: int,
        time_ms: float,
    ) -> None:
        """Report task execution result for fitness computation.

        Fitness = success_weight / (steps * time_weight)
        """
        success_score = 1.0 if success else 0.0
        step_penalty = steps / 30.0  # normalized by max_steps
        time_penalty = time_ms / 60000.0  # normalized by 1 minute

        fitness = success_score / max(step_penalty + time_penalty, 0.01)

        # Exponential moving average of fitness
        if genome.task_count == 0:
            genome.fitness = fitness
        else:
            genome.fitness = 0.7 * genome.fitness + 0.3 * fitness
        genome.task_count += 1

    def _tournament_select(self) -> StrategyGenome:
        """Select parent via tournament selection."""
        candidates = random.sample(
            self._population,
            min(self._tournament_size, len(self._population)),
        )
        return max(candidates, key=lambda g: g.fitness)

    def evolve(self) -> None:
        """Create next generation via selection, crossover, and mutation."""
        self._generation += 1

        # Sort by fitness
        self._population.sort(key=lambda g: g.fitness, reverse=True)

        new_pop: List[StrategyGenome] = []

        # Elitism: keep top individuals unchanged
        for i in range(min(self._elitism, len(self._population))):
            elite = deepcopy(self._population[i])
            elite.generation = self._generation
            new_pop.append(elite)

        # Fill rest via crossover + mutation
        while len(new_pop) < self._pop_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()

            child = crossover_blend(parent1, parent2)
            child.mutate(self._mutation_rate, self._mutation_strength)
            child.generation = self._generation
            child.fitness = 0.0
            child.task_count = 0
            new_pop.append(child)

        self._population = new_pop[:self._pop_size]

        best = max(self._population, key=lambda g: g.fitness)
        logger.info(
            f"Generation {self._generation}: "
            f"best_fitness={best.fitness:.4f}, "
            f"pop_size={len(self._population)}"
        )

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def best_genome(self) -> StrategyGenome:
        return max(self._population, key=lambda g: g.fitness)

    @property
    def population_stats(self) -> Dict[str, float]:
        fitnesses = [g.fitness for g in self._population]
        return {
            "generation": self._generation,
            "best": max(fitnesses),
            "worst": min(fitnesses),
            "mean": sum(fitnesses) / len(fitnesses),
            "evaluated": sum(1 for g in self._population if g.task_count > 0),
        }
