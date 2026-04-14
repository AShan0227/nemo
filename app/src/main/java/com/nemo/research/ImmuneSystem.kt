package com.nemo.research

import kotlin.math.sqrt
import kotlin.random.Random

/**
 * Negative Selection Algorithm (NSA) — anomaly detection.
 * Learns "normal" screen patterns, generates detectors for anomalies
 * (crash dialogs, wrong app, captcha). Ported from immune.py.
 */

data class ScreenFeatures(
    val elementCount: Int,
    val interactiveCount: Int,
    val textLength: Int,
    val hasInput: Boolean,
    val hasButton: Boolean,
    val hasScrollable: Boolean,
    val uniqueClasses: Int,
    val depthEstimate: Int,
) {
    fun toVector(): FloatArray = floatArrayOf(
        elementCount / 100f, interactiveCount / 50f, textLength / 500f,
        if (hasInput) 1f else 0f, if (hasButton) 1f else 0f,
        if (hasScrollable) 1f else 0f, uniqueClasses / 20f, depthEstimate / 10f,
    )
}

data class Detector(val center: FloatArray, val radius: Float) {
    fun matches(features: FloatArray): Boolean {
        var sum = 0f
        for (i in center.indices) { val d = center[i] - features[i]; sum += d * d }
        return sqrt(sum) <= radius
    }
}

data class AnomalyResult(val isAnomaly: Boolean, val matchingDetectors: Int, val confidence: Float, val message: String = "")

class ImmuneSystem(
    private val detectorCount: Int = 50,
    private val detectorRadius: Float = 0.3f,
    private val selfRadius: Float = 0.2f,
    private val anomalyThreshold: Int = 3,
) {
    private val selfSet = mutableListOf<FloatArray>()
    private val detectors = mutableListOf<Detector>()
    var trained = false; private set

    fun addSelfSample(features: ScreenFeatures) {
        selfSet.add(features.toVector())
        trained = false
    }

    fun train(maxCandidates: Int = 1000): Int {
        if (selfSet.isEmpty()) return 0
        val dim = selfSet[0].size
        detectors.clear()
        var tried = 0
        while (detectors.size < detectorCount && tried < maxCandidates) {
            val center = FloatArray(dim) { Random.nextFloat() * 1.5f }
            tried++
            val matchesSelf = selfSet.any { self ->
                var sum = 0f; for (i in center.indices) { val d = center[i] - self[i]; sum += d * d }
                sqrt(sum) < selfRadius
            }
            if (!matchesSelf) detectors.add(Detector(center, detectorRadius))
        }
        trained = true
        return detectors.size
    }

    fun check(features: ScreenFeatures): AnomalyResult {
        if (!trained) return AnomalyResult(false, 0, 0f, "Not trained")
        val vec = features.toVector()
        val matching = detectors.count { it.matches(vec) }
        val isAnomaly = matching >= anomalyThreshold
        return AnomalyResult(isAnomaly, matching, (matching.toFloat() / anomalyThreshold).coerceAtMost(1f),
            if (isAnomaly) "$matching detectors fired" else "Normal")
    }

    val selfSetSize: Int get() = selfSet.size
}
