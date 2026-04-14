package com.nemo.research

/**
 * Homeostatic regulation — negative feedback control loops.
 * Maintains agent health at setpoints, outputs behavioral adjustments.
 * Ported from homeostasis.py.
 */

data class Adjustment(val name: String, val description: String, val value: Float)

class RegulatedVariable(
    val name: String,
    val setpoint: Float,
    private val gain: Float = 1f,
    private val historyCapacity: Int = 50,
) {
    var current: Float = 0f; private set
    private val history = ArrayDeque<Float>(historyCapacity)

    val drive: Float get() = setpoint - current
    val normalizedDrive: Float get() = if (setpoint == 0f) 0f else (drive / setpoint).coerceIn(-1f, 1f)

    val trend: Float get() {
        if (history.size < 3) return 0f
        val recent = history.takeLast(3)
        return (recent.last() - recent.first()) / 2f
    }

    fun update(value: Float) {
        current = value
        if (history.size >= historyCapacity) history.removeFirst()
        history.addLast(value)
    }
}

class HomeostasisRegulator {
    val variables = mapOf(
        "success_rate" to RegulatedVariable("success_rate", 0.9f),
        "response_latency_ms" to RegulatedVariable("response_latency_ms", 2000f, gain = 0.5f),
        "error_rate" to RegulatedVariable("error_rate", 0.05f, gain = 1.5f),
    )

    fun updateMetrics(successRate: Float? = null, latencyMs: Float? = null, errorRate: Float? = null) {
        successRate?.let { variables["success_rate"]?.update(it) }
        latencyMs?.let { variables["response_latency_ms"]?.update(it) }
        errorRate?.let { variables["error_rate"]?.update(it) }
    }

    fun getAdjustments(): List<Adjustment> {
        val adj = mutableListOf<Adjustment>()
        val sr = variables["success_rate"]!!
        if (sr.drive > 0.1f) adj.add(Adjustment("verify_actions", "Add extra verify step", sr.drive * 1f))
        val lat = variables["response_latency_ms"]!!
        if (lat.drive < -500f) adj.add(Adjustment("reduce_prompt", "Filter to top-k elements", lat.normalizedDrive * 0.5f))
        val er = variables["error_rate"]!!
        if (er.drive < -0.05f) adj.add(Adjustment("increase_delay", "Increase action delay", (-er.drive) * 1.5f))
        return adj
    }
}
