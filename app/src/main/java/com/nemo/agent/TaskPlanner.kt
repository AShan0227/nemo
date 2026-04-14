package com.nemo.agent

import com.nemo.model.OnDeviceLLM
import com.nemo.model.PromptTemplates
import org.json.JSONArray
import org.json.JSONObject
import java.util.PriorityQueue

/**
 * Task planner with A* search and LLM fallback.
 * Ported from Python src/agent/planner.py.
 */
data class PlanStep(
    val stepNumber: Int,
    val actionType: String,
    val params: Map<String, Any> = emptyMap(),
    val description: String,
    val expectedState: String,
    val estimatedCost: Float,
)

data class TaskPlan(
    val task: String,
    val steps: List<PlanStep>,
    val totalCost: Float,
    val confidence: Float,
)

data class PlannerEdge(
    val toState: String,
    val actionType: String,
    var successCount: Int = 0,
    var attemptCount: Int = 0,
    var cost: Float = 1f,
) {
    val successRate: Float
        get() = if (attemptCount > 0) successCount.toFloat() / attemptCount else 0f
}

class PlannerGraph {
    private val edges: MutableMap<String, MutableList<PlannerEdge>> = mutableMapOf()

    fun recordTransition(from: String, to: String, actionType: String, success: Boolean) {
        val bucket = edges.getOrPut(from) { mutableListOf() }
        val edge = bucket.firstOrNull { it.toState == to && it.actionType == actionType }
            ?: PlannerEdge(toState = to, actionType = actionType).also { bucket.add(it) }

        edge.attemptCount += 1
        if (success) edge.successCount += 1

        val baseCost = 1f
        val successPenalty = if (edge.successRate > 0f) (1f - edge.successRate) else 1f
        edge.cost = baseCost + successPenalty
    }

    fun neighbors(state: String): List<PlannerEdge> = edges[state].orEmpty()

    fun aStar(startState: String, goalState: String): List<PlannerEdge>? {
        if (startState == goalState) return emptyList()

        data class Node(
            val state: String,
            val f: Float,
            val g: Float,
            val path: List<PlannerEdge>,
        )

        val queue = PriorityQueue<Node>(compareBy<Node> { it.f }.thenBy { it.g })
        val bestG = mutableMapOf(startState to 0f)
        queue.offer(Node(state = startState, f = 0f, g = 0f, path = emptyList()))

        while (queue.isNotEmpty()) {
            val current = queue.poll()
            if (current.state == goalState) return current.path
            if (current.g > (bestG[current.state] ?: Float.MAX_VALUE)) continue

            for (edge in neighbors(current.state)) {
                val newG = current.g + edge.cost
                if (newG >= (bestG[edge.toState] ?: Float.MAX_VALUE)) continue
                bestG[edge.toState] = newG

                val heuristic = 0f // no semantic state distance yet
                queue.offer(
                    Node(
                        state = edge.toState,
                        g = newG,
                        f = newG + heuristic,
                        path = current.path + edge,
                    )
                )
            }
        }

        return null
    }
}

class TaskPlanner(private val graph: PlannerGraph = PlannerGraph()) {

    fun recordTransition(from: String, to: String, actionType: String, success: Boolean) {
        graph.recordTransition(from, to, actionType, success)
    }

    fun planFromGraph(currentState: String, goalState: String): TaskPlan? {
        val path = graph.aStar(currentState, goalState) ?: return null
        val steps = mutableListOf<PlanStep>()
        var total = 0f

        path.forEachIndexed { index, edge ->
            steps += PlanStep(
                stepNumber = index + 1,
                actionType = edge.actionType,
                params = emptyMap(),
                description = "${edge.actionType} -> ${edge.toState}",
                expectedState = edge.toState,
                estimatedCost = edge.cost,
            )
            total += edge.cost
        }

        val avgSuccess = if (path.isNotEmpty()) path.map { it.successRate }.average().toFloat() else 1f
        return TaskPlan(
            task = "$currentState -> $goalState",
            steps = steps,
            totalCost = total,
            confidence = avgSuccess,
        )
    }

    suspend fun planWithLlm(
        llm: OnDeviceLLM,
        task: String,
        currentApp: String,
        screenSummary: String,
    ): TaskPlan? {
        val prompt = PromptTemplates.buildPlanningPrompt(
            task = task,
            currentApp = currentApp.ifBlank { "unknown" },
            screenSummary = screenSummary,
        )

        val raw = llm.generate(prompt)
        val array = extractJsonArray(raw) ?: return null

        val steps = mutableListOf<PlanStep>()
        for (i in 0 until array.length()) {
            val item = array.optJSONObject(i) ?: continue
            val action = item.optString("action_name").ifBlank { item.optString("action") }
            if (action.isBlank()) continue
            val params = jsonToMap(item.optJSONObject("params"))

            steps += PlanStep(
                stepNumber = i + 1,
                actionType = action,
                params = params,
                description = item.optString("action", action),
                expectedState = item.optString("expected_result", ""),
                estimatedCost = item.optDouble("estimated_cost", 1.0).toFloat(),
            )
        }

        if (steps.isEmpty()) return null

        return TaskPlan(
            task = task,
            steps = steps,
            totalCost = steps.sumOf { it.estimatedCost.toDouble() }.toFloat(),
            confidence = 0.5f,
        )
    }

    fun nextActionHint(plan: TaskPlan?, cursor: Int): PlanStep? {
        if (plan == null) return null
        if (cursor < 0 || cursor >= plan.steps.size) return null
        return plan.steps[cursor]
    }

    private fun extractJsonArray(raw: String): JSONArray? {
        val start = raw.indexOf('[')
        val end = raw.lastIndexOf(']')
        if (start < 0 || end <= start) return null
        return try {
            JSONArray(raw.substring(start, end + 1))
        } catch (_: Exception) {
            null
        }
    }

    private fun jsonToMap(json: JSONObject?): Map<String, Any> {
        if (json == null) return emptyMap()
        val map = linkedMapOf<String, Any>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val key = keys.next()
            map[key] = json.get(key)
        }
        return map
    }
}
