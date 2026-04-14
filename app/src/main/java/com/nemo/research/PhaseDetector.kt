package com.nemo.research

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Phase transition detector — detect capability jumps (grokking).
 * Sliding-window variance analysis. Ported from phase_detector.py.
 */

data class PerformanceSnapshot(
    val timestamp: Long,
    val successRate: Float,
    val avgSteps: Float,
    val graphNodes: Int,
    val graphEdges: Int,
)

data class PhaseTransition(
    val timestamp: Long,
    val metricName: String,
    val oldValue: Float,
    val newValue: Float,
    val changeRatio: Float,
)

class PhaseDetector(
    private val windowSize: Int = 10,
    private val baselineSize: Int = 50,
    private val sensitivity: Float = 2f,
) {
    private val history = mutableMapOf(
        "success_rate" to ArrayDeque<Float>(baselineSize),
        "avg_steps" to ArrayDeque<Float>(baselineSize),
        "graph_coverage" to ArrayDeque<Float>(baselineSize),
    )
    val transitions = mutableListOf<PhaseTransition>()

    fun record(snapshot: PerformanceSnapshot): PhaseTransition? {
        val sr = history["success_rate"]!!; sr.addLast(snapshot.successRate); if (sr.size > baselineSize) sr.removeFirst()
        val st = history["avg_steps"]!!; st.addLast(snapshot.avgSteps); if (st.size > baselineSize) st.removeFirst()
        val gc = history["graph_coverage"]!!; gc.addLast((snapshot.graphNodes + snapshot.graphEdges).toFloat()); if (gc.size > baselineSize) gc.removeFirst()

        if (sr.size < windowSize + 5) return null

        for ((name, values) in history) {
            val t = checkTransition(name, values.toList(), snapshot.timestamp) ?: continue
            transitions.add(t)
            return t
        }
        return null
    }

    private fun checkTransition(name: String, values: List<Float>, timestamp: Long): PhaseTransition? {
        if (values.size < windowSize + 5) return null
        val recent = values.takeLast(windowSize)
        val baseline = values.dropLast(windowSize)
        if (baseline.isEmpty()) return null

        val bMean = baseline.average().toFloat()
        val rMean = recent.average().toFloat()
        val variance = baseline.map { (it - bMean) * (it - bMean) }.average().toFloat()
        val std = sqrt(variance)

        if (std < 0.001f) {
            if (abs(rMean - bMean) > 0.05f) {
                return PhaseTransition(timestamp, name, bMean, rMean, rMean / bMean.coerceAtLeast(0.001f))
            }
            return null
        }

        val zScore = abs(rMean - bMean) / std
        val isImprovement = if (name == "avg_steps") rMean < bMean else rMean > bMean
        if (zScore > sensitivity && isImprovement) {
            return PhaseTransition(timestamp, name, bMean, rMean, rMean / bMean.coerceAtLeast(0.001f))
        }
        return null
    }
}
