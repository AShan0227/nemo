package com.nemo.model

import kotlin.math.ln
import kotlin.math.max

/**
 * Entropy-based routing for System 1 / 1.5 / 2 decision depth.
 * Ported from Python src/llm/router.py.
 */
enum class ReasoningDepth { SYSTEM_1, SYSTEM_1_5, SYSTEM_2 }

data class RoutingDecision(
    val depth: ReasoningDepth,
    val entropy: Float,
    val confidence: Float,
    val source: String = "unknown",
)

data class RouterStats(
    var system1Hits: Int = 0,
    var system15Hits: Int = 0,
    var system2Hits: Int = 0,
    var cacheHits: Int = 0,
    var cacheMisses: Int = 0,
    var realEntropyRoutes: Int = 0,
    var heuristicRoutes: Int = 0,
) {
    val total: Int
        get() = system1Hits + system15Hits + system2Hits

    val savingsRatio: Float
        get() = if (total > 0) (system1Hits + system15Hits).toFloat() / total else 0f
}

data class RoutingBenchmarkCase(
    val screenHash: String,
    val expectedDepth: ReasoningDepth,
    val actionProbs: List<Float>,
    val realEntropy: Float,
    val cached: Boolean = false,
)

data class RoutingAccuracyReport(
    val sampleSize: Int,
    val heuristicAccuracy: Float,
    val realEntropyAccuracy: Float,
) {
    val delta: Float
        get() = realEntropyAccuracy - heuristicAccuracy
}

class EntropyRouter(
    private val thresholdLow: Float = 0.3f,
    private val thresholdHigh: Float = 0.7f,
) {
    private val cache = mutableMapOf<String, String>() // screenHash -> actionJSON
    private val entropyMemory = mutableMapOf<String, Float>() // screenHash -> entropy

    val stats = RouterStats()

    fun computeEntropy(probs: List<Float>): Float {
        if (probs.isEmpty()) return 0f
        val log2 = ln(2f)
        var entropy = 0f
        for (p in probs) {
            if (p > 0f) entropy -= p * (ln(p) / log2)
        }
        val maxEntropy = ln(max(probs.size, 2).toFloat()) / log2
        return if (maxEntropy > 0f) entropy / maxEntropy else 0f
    }

    fun observeEntropy(screenHash: String, entropy: Float) {
        if (!entropy.isFinite()) return
        entropyMemory[screenHash] = entropy.coerceIn(0f, 1f)
    }

    fun getObservedEntropy(screenHash: String): Float? = entropyMemory[screenHash]

    fun route(
        screenHash: String,
        actionProbs: List<Float>? = null,
        entropy: Float? = null,
    ): RoutingDecision {
        val (resolvedEntropy, source) = when {
            entropy != null -> {
                stats.realEntropyRoutes += 1
                entropy.coerceIn(0f, 1f) to "real_entropy"
            }

            entropyMemory.containsKey(screenHash) -> {
                stats.realEntropyRoutes += 1
                entropyMemory.getValue(screenHash) to "observed_entropy"
            }

            actionProbs != null -> {
                stats.heuristicRoutes += 1
                computeEntropy(actionProbs).coerceIn(0f, 1f) to "heuristic_probs"
            }

            else -> {
                stats.heuristicRoutes += 1
                1f to "cold_start"
            }
        }

        val confidence = 1f - resolvedEntropy
        val depth = depthFromEntropy(screenHash, resolvedEntropy)
        return RoutingDecision(
            depth = depth,
            entropy = resolvedEntropy,
            confidence = confidence,
            source = source,
        )
    }

    private fun depthFromEntropy(screenHash: String, entropy: Float): ReasoningDepth {
        return when {
            entropy < thresholdLow && cache.containsKey(screenHash) -> {
                stats.system1Hits += 1
                stats.cacheHits += 1
                ReasoningDepth.SYSTEM_1
            }

            entropy < thresholdHigh -> {
                stats.system15Hits += 1
                if (!cache.containsKey(screenHash)) stats.cacheMisses += 1
                ReasoningDepth.SYSTEM_1_5
            }

            else -> {
                stats.system2Hits += 1
                ReasoningDepth.SYSTEM_2
            }
        }
    }

    fun cacheAction(screenHash: String, action: String) {
        cache[screenHash] = action
    }

    fun getCachedAction(screenHash: String): String? = cache[screenHash]
}

fun compareRoutingAccuracy(
    cases: List<RoutingBenchmarkCase>,
    thresholdLow: Float = 0.3f,
    thresholdHigh: Float = 0.7f,
): RoutingAccuracyReport {
    if (cases.isEmpty()) {
        return RoutingAccuracyReport(
            sampleSize = 0,
            heuristicAccuracy = 0f,
            realEntropyAccuracy = 0f,
        )
    }

    val heuristicRouter = EntropyRouter(thresholdLow, thresholdHigh)
    val realRouter = EntropyRouter(thresholdLow, thresholdHigh)

    var heuristicHits = 0
    var realHits = 0

    for (c in cases) {
        if (c.cached) {
            heuristicRouter.cacheAction(c.screenHash, "{\"action\":\"tap\"}")
            realRouter.cacheAction(c.screenHash, "{\"action\":\"tap\"}")
        }

        val heuristic = heuristicRouter.route(c.screenHash, actionProbs = c.actionProbs)
        val real = realRouter.route(c.screenHash, entropy = c.realEntropy)

        if (heuristic.depth == c.expectedDepth) heuristicHits += 1
        if (real.depth == c.expectedDepth) realHits += 1
    }

    val size = cases.size.toFloat()
    return RoutingAccuracyReport(
        sampleSize = cases.size,
        heuristicAccuracy = heuristicHits / size,
        realEntropyAccuracy = realHits / size,
    )
}
