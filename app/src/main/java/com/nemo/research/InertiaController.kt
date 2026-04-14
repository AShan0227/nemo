package com.nemo.research

import kotlin.math.max
import kotlin.math.min

/**
 * Friction/Inertia — plan stability controller.
 * Prevents oscillation between plans mid-execution.
 * Ported from inertia.py.
 */

data class PlanCommitment(
    val plannedActions: List<String>,
    var currentStep: Int = 0,
    var deviations: Int = 0,
    var totalStepsExecuted: Int = 0,
) {
    val progress: Float get() = if (plannedActions.isEmpty()) 0f else min(currentStep.toFloat() / plannedActions.size, 1f)
    val deviationRate: Float get() = if (totalStepsExecuted == 0) 0f else deviations.toFloat() / totalStepsExecuted
    val nextPlannedAction: String? get() = plannedActions.getOrNull(currentStep)
}

data class InertiaDecision(val usePlanned: Boolean, val reason: String, val inertiaScore: Float)

class InertiaController(
    private val baseInertia: Float = 0.3f,
    private val progressWeight: Float = 0.5f,
    private val maxInertia: Float = 0.8f,
) {
    private var commitment: PlanCommitment? = null

    val hasPlan: Boolean get() = commitment?.nextPlannedAction != null

    fun setPlan(actions: List<String>) { commitment = PlanCommitment(actions) }
    fun clearPlan() { commitment = null }

    fun computeInertia(): Float {
        val c = commitment ?: return 0f
        val inertia = baseInertia + c.progress * progressWeight - c.deviationRate * 0.5f
        return max(0f, min(inertia, maxInertia))
    }

    fun shouldFollowPlan(planned: String, newAction: String, newConfidence: Float = 0.5f): InertiaDecision {
        if (planned == newAction) return InertiaDecision(true, "LLM agrees with plan", 0f)
        val inertia = computeInertia()
        return if (newConfidence > inertia) {
            commitment?.deviations = (commitment?.deviations ?: 0) + 1
            InertiaDecision(false, "Override (conf=${"%.2f".format(newConfidence)} > inertia=${"%.2f".format(inertia)})", inertia)
        } else {
            InertiaDecision(true, "Inertia holds (conf=${"%.2f".format(newConfidence)} < inertia=${"%.2f".format(inertia)})", inertia)
        }
    }

    fun advance() { commitment?.let { it.currentStep++; it.totalStepsExecuted++ } }
    fun getNextPlannedAction(): String? = commitment?.nextPlannedAction
}
