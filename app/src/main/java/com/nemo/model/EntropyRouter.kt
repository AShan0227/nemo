package com.nemo.model

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min

/**
 * Entropy-based routing — System 1 / 1.5 / 2 decision depth.
 * Ported from Python research/router.py.
 *
 * On-device this is crucial: simple actions use cache (0ms),
 * complex decisions activate Gemma (~100-500ms).
 */
enum class ReasoningDepth { SYSTEM_1, SYSTEM_1_5, SYSTEM_2 }

data class RoutingDecision(
    val depth: ReasoningDepth,
    val entropy: Float,
    val confidence: Float,
)

class EntropyRouter(
    private val thresholdLow: Float = 0.3f,
    private val thresholdHigh: Float = 0.7f,
) {
    private val cache = mutableMapOf<String, String>() // screenHash → actionJSON
    var system1Hits = 0; private set
    var system15Hits = 0; private set
    var system2Hits = 0; private set

    fun computeEntropy(probs: List<Float>): Float {
        if (probs.isEmpty()) return 0f
        val log2 = ln(2.0f)
        var entropy = 0f
        for (p in probs) {
            if (p > 0f) entropy -= p * (ln(p) / log2)
        }
        val maxEntropy = ln(max(probs.size, 2).toFloat()) / log2
        return if (maxEntropy > 0f) entropy / maxEntropy else 0f
    }

    fun route(screenHash: String, actionProbs: List<Float>? = null): RoutingDecision {
        val entropy = actionProbs?.let { computeEntropy(it) } ?: 1f
        val confidence = 1f - entropy

        val depth = when {
            entropy < thresholdLow && screenHash in cache -> {
                system1Hits++; ReasoningDepth.SYSTEM_1
            }
            entropy < thresholdHigh -> {
                system15Hits++; ReasoningDepth.SYSTEM_1_5
            }
            else -> {
                system2Hits++; ReasoningDepth.SYSTEM_2
            }
        }
        return RoutingDecision(depth, entropy, confidence)
    }

    fun cacheAction(screenHash: String, action: String) { cache[screenHash] = action }
    fun getCachedAction(screenHash: String): String? = cache[screenHash]

    val savingsRatio: Float
        get() {
            val total = system1Hits + system15Hits + system2Hits
            return if (total > 0) (system1Hits + system15Hits).toFloat() / total else 0f
        }
}
