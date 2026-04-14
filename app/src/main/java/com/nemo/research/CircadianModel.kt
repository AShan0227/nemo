package com.nemo.research

import java.util.Calendar

/**
 * Circadian rhythm modeling — time-aware behavior adaptation.
 * 24x7 time-slot activity tracking + behavior modifiers.
 * Ported from circadian.py.
 */

data class TimeSlot(
    val hour: Int,
    val dayOfWeek: Int,
    val appCounts: MutableMap<String, Int> = mutableMapOf(),
    var totalTasks: Int = 0,
    var avgDurationMs: Float = 0f,
) {
    fun topApps(k: Int = 3): List<Pair<String, Int>> =
        appCounts.entries.sortedByDescending { it.value }.take(k).map { it.key to it.value }
}

data class BehaviorModifiers(
    val actionDelayMultiplier: Float = 1f,
    val verificationLevel: Float = 1f,
    val proactiveSuggestions: Float = 0f,
)

class CircadianModel {
    private val slots = mutableMapOf<Pair<Int, Int>, TimeSlot>()

    private fun getSlot(hour: Int, dow: Int): TimeSlot =
        slots.getOrPut(hour to dow) { TimeSlot(hour, dow) }

    fun recordActivity(appPackage: String, actionType: String, durationMs: Float = 0f, timeMs: Long? = null) {
        val cal = Calendar.getInstance().apply { timeMs?.let { timeInMillis = it } }
        val slot = getSlot(cal.get(Calendar.HOUR_OF_DAY), cal.get(Calendar.DAY_OF_WEEK) - 1)
        slot.appCounts[appPackage] = (slot.appCounts[appPackage] ?: 0) + 1
        slot.totalTasks++
        if (durationMs > 0) {
            slot.avgDurationMs = if (slot.avgDurationMs == 0f) durationMs else 0.8f * slot.avgDurationMs + 0.2f * durationMs
        }
    }

    fun predictApps(timeMs: Long? = null, topK: Int = 3): List<String> {
        val cal = Calendar.getInstance().apply { timeMs?.let { timeInMillis = it } }
        val slot = slots[cal.get(Calendar.HOUR_OF_DAY) to (cal.get(Calendar.DAY_OF_WEEK) - 1)]
        return slot?.topApps(topK)?.map { it.first } ?: emptyList()
    }

    fun getBehaviorModifier(timeMs: Long? = null): BehaviorModifiers {
        val cal = Calendar.getInstance().apply { timeMs?.let { timeInMillis = it } }
        return when (cal.get(Calendar.HOUR_OF_DAY)) {
            in 0..6 -> BehaviorModifiers(1.5f, 0.5f, 0f)
            in 7..8 -> BehaviorModifiers(0.8f, 1f, 0.8f)
            in 9..17 -> BehaviorModifiers(1f, 1f, 0.5f)
            in 18..21 -> BehaviorModifiers(1.1f, 0.8f, 0.6f)
            else -> BehaviorModifiers(1.3f, 0.7f, 0.2f)
        }
    }

    val totalActivities: Int get() = slots.values.sumOf { it.totalTasks }
}
