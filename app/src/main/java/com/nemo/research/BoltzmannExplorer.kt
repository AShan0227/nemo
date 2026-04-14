package com.nemo.research

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.random.Random

/**
 * Boltzmann action selection — entropy-regularized exploration.
 * P(a|s) ∝ exp(Q(s,a)/τ). Temperature anneals over time.
 * Ported from explorer.py.
 */

data class ActionValue(val actionName: String, var qValue: Float = 0f, var visitCount: Int = 0)

class BoltzmannExplorer(
    initialTemperature: Float = 1f,
    private val minTemperature: Float = 0.1f,
    private val annealingRate: Float = 0.995f,
    private val entropyBonus: Float = 0.1f,
) {
    var temperature: Float = initialTemperature; private set
    private val qTable = mutableMapOf<String, MutableList<ActionValue>>()
    var totalSteps: Int = 0; private set

    fun selectAction(stateHash: String, actions: List<String>): Pair<String, Float> {
        if (actions.isEmpty()) return "wait" to 1f
        val values = qTable[stateHash] ?: emptyList()
        val valueMap = values.associate { it.actionName to it.qValue }

        val logits = actions.map { a ->
            val q = valueMap[a] ?: 0f
            val visits = values.firstOrNull { it.actionName == a }?.visitCount ?: 0
            val bonus = entropyBonus / max(visits.toFloat(), 1f).let { kotlin.math.sqrt(it) }
            (q + bonus) / max(temperature, 0.01f)
        }

        val maxLogit = logits.max()
        val expLogits = logits.map { exp(it - maxLogit) }
        val total = expLogits.sum()
        val probs = expLogits.map { it / total }

        var r = Random.nextFloat()
        for (i in probs.indices) {
            r -= probs[i]
            if (r <= 0) { totalSteps++; return actions[i] to probs[i] }
        }
        totalSteps++
        return actions.last() to probs.last()
    }

    fun updateValue(stateHash: String, actionName: String, reward: Float, lr: Float = 0.1f) {
        val values = qTable.getOrPut(stateHash) { mutableListOf() }
        val existing = values.firstOrNull { it.actionName == actionName }
        if (existing != null) {
            existing.qValue += lr * (reward - existing.qValue)
            existing.visitCount++
        } else {
            values.add(ActionValue(actionName, reward, 1))
        }
    }

    fun anneal() { temperature = max(minTemperature, temperature * annealingRate) }

    val knownStates: Int get() = qTable.size
}
