package com.nemo.research

import kotlin.math.max
import kotlin.random.Random

/**
 * Genetic algorithm — evolve agent execution strategies.
 * Tournament selection + BLX-α crossover + Gaussian mutation.
 * Ported from evolution.py.
 */

data class StrategyGenome(
    val genes: MutableMap<String, Float>,
    var fitness: Float = 0f,
    var taskCount: Int = 0,
    var generation: Int = 0,
) {
    fun mutate(rate: Float = 0.1f, strength: Float = 0.2f) {
        for (key in genes.keys.toList()) {
            if (Random.nextFloat() < rate) {
                val delta = genes[key]!! * strength * Random.nextGaussian().toFloat()
                genes[key] = max(0.01f, genes[key]!! + delta)
            }
        }
    }

    fun get(key: String, default: Float = 0f): Float = genes[key] ?: default

    companion object {
        fun default(): StrategyGenome = StrategyGenome(mutableMapOf(
            "action_delay_ms" to 500f, "scroll_amount" to 0.25f,
            "max_retries" to 2f, "verify_after_action" to 1f,
            "exploration_rate" to 0.1f, "prompt_detail_level" to 1f,
            "safety_strictness" to 1f, "focus_tap_delay_ms" to 300f,
        ))

        fun random(): StrategyGenome = default().also { g ->
            g.genes.keys.toList().forEach { k -> g.genes[k] = g.genes[k]!! * Random.nextFloat() * 1.7f + 0.3f }
        }
    }
}

fun crossoverBlend(p1: StrategyGenome, p2: StrategyGenome, alpha: Float = 0.5f): StrategyGenome {
    val genes = mutableMapOf<String, Float>()
    for (key in p1.genes.keys) {
        val v1 = p1.genes[key]!!; val v2 = p2.genes[key]!!
        val low = minOf(v1, v2) - alpha * kotlin.math.abs(v2 - v1)
        val high = maxOf(v1, v2) + alpha * kotlin.math.abs(v2 - v1)
        genes[key] = max(0.01f, Random.nextFloat() * (high - low) + low)
    }
    return StrategyGenome(genes)
}

class StrategyEvolver(
    private val populationSize: Int = 20,
    private val mutationRate: Float = 0.15f,
    private val mutationStrength: Float = 0.2f,
    private val elitismCount: Int = 2,
    private val tournamentSize: Int = 3,
) {
    private var population = MutableList(populationSize) { if (it == 0) StrategyGenome.default() else StrategyGenome.random() }
    var generation: Int = 0; private set

    fun getStrategy(): StrategyGenome {
        val evaluated = population.filter { it.taskCount > 0 }
        return if (evaluated.isNotEmpty()) evaluated.maxByOrNull { it.fitness }!! else population.random()
    }

    fun reportFitness(genome: StrategyGenome, success: Boolean, steps: Int, timeMs: Float) {
        val score = (if (success) 1f else 0f) / max(steps / 30f + timeMs / 60000f, 0.01f)
        genome.fitness = if (genome.taskCount == 0) score else 0.7f * genome.fitness + 0.3f * score
        genome.taskCount++
    }

    fun evolve() {
        generation++
        population.sortByDescending { it.fitness }
        val newPop = mutableListOf<StrategyGenome>()
        for (i in 0 until minOf(elitismCount, population.size)) {
            val elite = StrategyGenome(population[i].genes.toMutableMap(), population[i].fitness, population[i].taskCount, generation)
            newPop.add(elite)
        }
        while (newPop.size < populationSize) {
            val p1 = tournamentSelect(); val p2 = tournamentSelect()
            val child = crossoverBlend(p1, p2)
            child.mutate(mutationRate, mutationStrength)
            child.generation = generation
            newPop.add(child)
        }
        population = newPop.take(populationSize).toMutableList()
    }

    private fun tournamentSelect(): StrategyGenome =
        population.shuffled().take(minOf(tournamentSize, population.size)).maxByOrNull { it.fitness }!!

    val bestGenome: StrategyGenome get() = population.maxByOrNull { it.fitness }!!
}
