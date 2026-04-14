package com.nemo.screen

import kotlin.math.max
import kotlin.math.min

// ---------------------------------------------------------------------------
// Core Dempster-Shafer evidence theory
// ---------------------------------------------------------------------------

/**
 * Mass function for Dempster-Shafer fusion.
 * Maps hypothesis labels to belief mass (normalized to sum = 1.0).
 */
data class EvidenceMass(val beliefs: Map<String, Double>) {
    companion object {
        fun normalized(raw: Map<String, Double>): EvidenceMass {
            val total = raw.values.sum()
            return if (total > 0) {
                EvidenceMass(raw.mapValues { it.value / total })
            } else {
                EvidenceMass(mapOf("unknown" to 1.0))
            }
        }

        fun unknown(): EvidenceMass = EvidenceMass(mapOf("unknown" to 1.0))
    }
}

data class CombineResult(val mass: EvidenceMass, val conflict: Double)

/**
 * Combine two independent evidence sources using Dempster's rule.
 */
fun dempsterCombine(m1: EvidenceMass, m2: EvidenceMass): CombineResult {
    val combined = mutableMapOf<String, Double>()
    var conflict = 0.0

    for ((h1, v1) in m1.beliefs) {
        for ((h2, v2) in m2.beliefs) {
            val product = v1 * v2
            if (h1 == h2 || h1 == "unknown" || h2 == "unknown") {
                val key = if (h1 == "unknown") h2 else h1
                combined[key] = (combined[key] ?: 0.0) + product
            } else {
                conflict += product
            }
        }
    }

    val normalization = 1.0 - conflict
    val normalizedBeliefs = if (normalization > 0) {
        combined.mapValues { it.value / normalization }
    } else {
        combined
    }

    return CombineResult(EvidenceMass.normalized(normalizedBeliefs), conflict)
}

/**
 * Sequentially fuse multiple evidence sources.
 * Returns combined mass and maximum conflict encountered.
 */
fun fuseMultiple(vararg masses: EvidenceMass): CombineResult {
    if (masses.isEmpty()) return CombineResult(EvidenceMass.unknown(), 0.0)

    var result = masses[0]
    var maxConflict = 0.0

    for (i in 1 until masses.size) {
        val (combined, conflict) = dempsterCombine(result, masses[i])
        result = combined
        maxConflict = max(maxConflict, conflict)
    }

    return CombineResult(result, maxConflict)
}

// ---------------------------------------------------------------------------
// Signal candidates (multi-modal inputs)
// ---------------------------------------------------------------------------

data class SignalCandidate(
    val label: String,
    val confidence: Double,
    val source: String, // "ui", "ocr", "visual"
    val bounds: IntArray? = null // [left, top, right, bottom]
) {
    fun normalizedLabel(): String = label.trim().lowercase()

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SignalCandidate) return false
        return label == other.label && confidence == other.confidence && source == other.source
    }

    override fun hashCode(): Int = label.hashCode() * 31 + source.hashCode()
}

data class FusionResult(
    val beliefs: Map<String, Double>,
    val ranked: List<Pair<String, Double>>,
    val conflict: Double,
    val sourceMasses: Map<String, EvidenceMass>
) {
    val bestLabel: String get() = ranked.firstOrNull()?.first ?: "unknown"
}

// ---------------------------------------------------------------------------
// Candidate generation from each modality
// ---------------------------------------------------------------------------

private fun clamp(value: Double, low: Double, high: Double) = max(low, min(high, value))

/**
 * Generate candidates from UI hierarchy elements.
 */
fun candidatesFromUi(
    elements: List<UIElement>,
    interactiveOnly: Boolean = true
): List<SignalCandidate> {
    return elements.mapNotNull { elem ->
        if (interactiveOnly && !(elem.clickable || elem.scrollable || elem.editable)) return@mapNotNull null

        val (label, baseConf) = when {
            elem.text.isNotEmpty() -> elem.text to 0.88
            elem.contentDesc.isNotEmpty() -> elem.contentDesc to 0.82
            elem.resourceId.isNotEmpty() -> elem.resourceId.substringAfterLast("/") to 0.72
            elem.className.isNotEmpty() -> elem.className.substringAfterLast(".") to 0.45
            else -> return@mapNotNull null
        }

        val boost = if (elem.clickable || elem.editable) 0.04 else 0.0
        val penalty = if (elem.scrollable) 0.08 else 0.0
        val confidence = clamp(baseConf + boost - penalty, 0.15, 0.99)

        SignalCandidate(
            label = label,
            confidence = confidence,
            source = "ui",
            bounds = intArrayOf(elem.bounds.left, elem.bounds.top, elem.bounds.right, elem.bounds.bottom),
        )
    }
}

/**
 * Generate candidates from OCR text blocks.
 */
fun candidatesFromOcr(
    blocks: List<OCREngine.TextBlock>,
    minConfidence: Double = 0.4
): List<SignalCandidate> {
    return blocks.filter { it.confidence >= minConfidence }.map { block ->
        SignalCandidate(
            label = block.text,
            confidence = clamp(block.confidence.toDouble(), 0.0, 1.0),
            source = "ocr",
            bounds = block.bounds
        )
    }
}

/**
 * Generate candidates from visual model labels.
 */
fun candidatesFromVisual(labels: Map<String, Double>): List<SignalCandidate> {
    return labels.filter { it.value > 0 }.map { (label, score) ->
        SignalCandidate(
            label = label,
            confidence = clamp(score, 0.0, 1.0),
            source = "visual"
        )
    }
}

// ---------------------------------------------------------------------------
// Fusion orchestration
// ---------------------------------------------------------------------------

/**
 * Convert candidates into an EvidenceMass.
 */
fun candidatesToMass(candidates: List<SignalCandidate>, minUnknown: Double = 0.1): EvidenceMass {
    if (candidates.isEmpty()) return EvidenceMass.unknown()

    val beliefs = mutableMapOf<String, Double>()
    var maxConf = 0.0

    for (c in candidates) {
        val label = c.normalizedLabel()
        if (label.isEmpty()) continue
        val score = clamp(c.confidence, 0.0, 1.0)
        beliefs[label] = max(beliefs[label] ?: 0.0, score)
        maxConf = max(maxConf, score)
    }

    if (beliefs.isEmpty()) return EvidenceMass.unknown()
    beliefs["unknown"] = max(minUnknown, 1.0 - maxConf)
    return EvidenceMass.normalized(beliefs)
}

/**
 * Fuse UI/OCR/Visual candidates into final ranked hypotheses.
 */
fun fuseScreenSources(
    uiCandidates: List<SignalCandidate>? = null,
    ocrCandidates: List<SignalCandidate>? = null,
    visualCandidates: List<SignalCandidate>? = null,
    topK: Int = 5
): FusionResult {
    val sourceCandidates = mapOf(
        "ui" to (uiCandidates ?: emptyList()),
        "ocr" to (ocrCandidates ?: emptyList()),
        "visual" to (visualCandidates ?: emptyList()),
    )

    val sourceMasses = mutableMapOf<String, EvidenceMass>()
    val masses = mutableListOf<EvidenceMass>()

    for ((source, candidates) in sourceCandidates) {
        if (candidates.isEmpty()) continue
        val mass = candidatesToMass(candidates)
        sourceMasses[source] = mass
        masses.add(mass)
    }

    if (masses.isEmpty()) {
        return FusionResult(
            beliefs = mapOf("unknown" to 1.0),
            ranked = listOf("unknown" to 1.0),
            conflict = 0.0,
            sourceMasses = emptyMap()
        )
    }

    val (combined, conflict) = fuseMultiple(*masses.toTypedArray())
    val ranked = combined.beliefs.entries
        .sortedByDescending { it.value }
        .take(topK)
        .map { it.key to it.value }

    return FusionResult(
        beliefs = combined.beliefs,
        ranked = ranked,
        conflict = conflict,
        sourceMasses = sourceMasses
    )
}
