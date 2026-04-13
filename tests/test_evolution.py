"""Tests for genetic algorithm strategy evolution."""

from src.research.evolution import (
    StrategyGenome, StrategyEvolver,
    crossover_uniform, crossover_blend,
)


def test_default_genome():
    g = StrategyGenome.default()
    assert g.genes["action_delay_ms"] == 500.0
    assert g.fitness == 0.0


def test_random_genome():
    g = StrategyGenome.random()
    # Should differ from default
    d = StrategyGenome.default()
    differs = any(g.genes[k] != d.genes[k] for k in g.genes)
    assert differs


def test_mutation():
    g = StrategyGenome.default()
    original = dict(g.genes)
    g.mutate(rate=1.0, strength=0.5)  # mutate every gene
    changed = any(g.genes[k] != original[k] for k in g.genes)
    assert changed


def test_uniform_crossover():
    p1 = StrategyGenome(genes={"a": 1.0, "b": 2.0, "c": 3.0})
    p2 = StrategyGenome(genes={"a": 10.0, "b": 20.0, "c": 30.0})
    c1, c2 = crossover_uniform(p1, p2)
    # Children should have values from either parent
    for key in p1.genes:
        assert c1.genes[key] in (p1.genes[key], p2.genes[key])


def test_blend_crossover():
    p1 = StrategyGenome(genes={"a": 1.0, "b": 2.0})
    p2 = StrategyGenome(genes={"a": 3.0, "b": 4.0})
    child = crossover_blend(p1, p2, alpha=0.0)
    # With alpha=0, child should be within parent range
    assert 1.0 <= child.genes["a"] <= 3.0
    assert 2.0 <= child.genes["b"] <= 4.0


def test_evolver_initial_population():
    evolver = StrategyEvolver(population_size=10)
    assert len(evolver._population) == 10


def test_fitness_reporting():
    evolver = StrategyEvolver(population_size=5)
    genome = evolver._population[0]
    evolver.report_fitness(genome, success=True, steps=5, time_ms=3000)
    assert genome.fitness > 0
    assert genome.task_count == 1


def test_evolution_cycle():
    evolver = StrategyEvolver(population_size=10)

    # Evaluate some genomes
    for g in evolver._population[:5]:
        evolver.report_fitness(g, success=True, steps=8, time_ms=5000)

    evolver.evolve()
    assert evolver.generation == 1
    assert len(evolver._population) == 10


def test_population_stats():
    evolver = StrategyEvolver(population_size=5)
    stats = evolver.population_stats
    assert stats["generation"] == 0
    assert stats["evaluated"] == 0
