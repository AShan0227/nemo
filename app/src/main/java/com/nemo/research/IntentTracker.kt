package com.nemo.research

import kotlin.math.ln
import kotlin.math.max

/**
 * Intent tracker — particle filter for ambiguous commands.
 * Multi-hypothesis tracking, collapses when evidence accumulates.
 * Ported from intent_tracker.py.
 */

data class IntentHypothesis(val description: String, var weight: Float = 1f, var evidenceFor: Int = 0, var evidenceAgainst: Int = 0)

data class IntentState(val collapsed: Boolean, val dominant: IntentHypothesis?, val hypotheses: List<IntentHypothesis>, val entropy: Float)

class IntentTracker(
    hypotheses: List<String>,
    private val collapseThreshold: Float = 0.8f,
    private val maxStepsBeforeAsk: Int = 5,
) {
    private val hypotheses = hypotheses.map { IntentHypothesis(it, 1f / hypotheses.size) }.toMutableList()
    private var stepCount = 0

    val isCollapsed: Boolean get() {
        if (hypotheses.isEmpty()) return true
        return hypotheses.maxOf { it.weight } >= collapseThreshold
    }

    val shouldAskUser: Boolean get() = stepCount >= maxStepsBeforeAsk && !isCollapsed

    fun updateEvidence(screenText: String) {
        stepCount++
        val lower = screenText.lowercase()
        for (h in hypotheses) {
            val keywords = h.description.lowercase().split(" ").toSet()
            val matches = keywords.count { it in lower }
            val ratio = if (keywords.isNotEmpty()) matches.toFloat() / keywords.size else 0f
            if (ratio > 0.3f) { h.weight *= 1f + ratio; h.evidenceFor++ }
            else if (ratio == 0f) { h.weight *= 0.7f; h.evidenceAgainst++ }
        }
        normalize()
    }

    private fun normalize() {
        val total = hypotheses.sumOf { it.weight.toDouble() }.toFloat()
        if (total > 0) hypotheses.forEach { it.weight /= total }
    }

    fun getBestIntent(): String = hypotheses.maxByOrNull { it.weight }?.description ?: ""

    fun getState(): IntentState {
        if (hypotheses.isEmpty()) return IntentState(true, null, emptyList(), 0f)
        val log2 = ln(2f)
        var entropy = 0f
        for (h in hypotheses) { if (h.weight > 0) entropy -= h.weight * (ln(h.weight) / log2) }
        val maxE = ln(max(hypotheses.size, 2).toFloat()) / log2
        val normEntropy = if (maxE > 0) entropy / maxE else 0f
        val best = hypotheses.maxByOrNull { it.weight }
        return IntentState(isCollapsed, if (isCollapsed) best else null, hypotheses.toList(), normEntropy)
    }
}
