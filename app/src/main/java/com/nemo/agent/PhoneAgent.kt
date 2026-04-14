package com.nemo.agent

import com.nemo.model.EntropyRouter
import com.nemo.model.OnDeviceLLM
import com.nemo.model.ReasoningDepth
import com.nemo.safety.SafetyContext
import com.nemo.safety.SafetyLayer
import com.nemo.safety.Verdict
import com.nemo.screen.ScreenState
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.delay
import org.json.JSONObject

/**
 * PhoneAgent — the main agent loop on Android.
 * Ported from Python agent.py with on-device execution.
 *
 * observe → route → think → safety → act → learn
 *
 * Key difference from Python version:
 * - Uses AccessibilityService instead of ADB (zero latency)
 * - Uses on-device Gemma instead of remote API (~100ms vs ~3s)
 * - All research mechanisms run locally
 */

enum class TaskStatus { PENDING, RUNNING, COMPLETED, FAILED, BLOCKED }

data class StepRecord(
    val step: Int,
    val action: String,
    val params: Map<String, Any> = emptyMap(),
    val reasoning: String = "",
    val success: Boolean = false,
    val durationMs: Long = 0,
    val error: String = "",
)

data class TaskResult(
    val status: TaskStatus,
    val summary: String = "",
    val steps: List<StepRecord> = emptyList(),
    val totalSteps: Int = 0,
)

class PhoneAgent(
    private val llm: OnDeviceLLM,
    private val maxSteps: Int = 30,
    private val actionDelayMs: Long = 300,
    private val safetyEnabled: Boolean = true,
) {
    private val router = EntropyRouter()
    private val safety = if (safetyEnabled) SafetyLayer() else null
    private var stepCount = 0

    private val service: NemoAccessibilityService?
        get() = NemoAccessibilityService.instance

    private val systemPrompt = """
You are an AI phone assistant controlling an Android device.
You observe the current screen (UI elements with indices) and decide the next action.

Available actions:
- tap(index): Tap element at index
- type_text(index, text): Focus element and type text
- scroll(direction): Scroll "up" or "down"
- back(): Press back
- home(): Go to home screen
- done(summary): Task complete

Rules:
1. ONLY use element indices from the current screen.
2. Explain reasoning before choosing action.
3. NEVER perform financial transactions or grant permissions.

Respond in JSON: {"reasoning":"...","action":"...","params":{...}}
""".trimIndent()

    suspend fun execute(task: String, onStep: ((StepRecord) -> Unit)? = null): TaskResult {
        stepCount = 0
        val steps = mutableListOf<StepRecord>()
        val svc = service ?: return TaskResult(TaskStatus.FAILED, "AccessibilityService not connected")

        while (stepCount < maxSteps) {
            stepCount++
            val startTime = System.currentTimeMillis()

            // 1. Observe
            val screen = svc.getScreenState()
            val screenHash = screen.computeHash()
            val screenContext = screen.toPromptStr()

            // 2. Route
            val routing = router.route(screenHash)
            var actionJson: String? = null

            if (routing.depth == ReasoningDepth.SYSTEM_1) {
                actionJson = router.getCachedAction(screenHash)
            }

            // 3. Think (on-device Gemma)
            if (actionJson == null) {
                actionJson = llm.decideAction(systemPrompt, screenContext, task)
            }

            // 4. Parse
            val decision = try {
                parseDecision(actionJson)
            } catch (e: Exception) {
                val record = StepRecord(stepCount, "error", error = e.message ?: "parse error")
                steps.add(record)
                onStep?.invoke(record)
                continue
            }

            val actionName = decision.optString("action", "")
            val reasoning = decision.optString("reasoning", "")
            val params = parseParams(decision.optJSONObject("params"))

            // Done check
            if (actionName == "done") {
                val record = StepRecord(stepCount, "done", params, reasoning, true)
                steps.add(record)
                onStep?.invoke(record)
                return TaskResult(TaskStatus.COMPLETED, params["summary"]?.toString() ?: reasoning, steps, stepCount)
            }

            // 5. Safety
            if (safety != null) {
                val ctx = SafetyContext(screenContext, stepCount, maxSteps, actionName)
                val result = safety.check(ctx)
                if (result.verdict == Verdict.BLOCK) {
                    val record = StepRecord(stepCount, actionName, params, reasoning, false, error = "Safety: ${result.message}")
                    steps.add(record)
                    onStep?.invoke(record)
                    return TaskResult(TaskStatus.BLOCKED, result.message, steps, stepCount)
                }
            }

            // 6. Execute
            val success = executeAction(svc, actionName, params, screen)
            val duration = System.currentTimeMillis() - startTime

            // 7. Cache successful actions
            if (success) {
                router.cacheAction(screenHash, actionJson)
            }

            val record = StepRecord(stepCount, actionName, params, reasoning, success, duration)
            steps.add(record)
            onStep?.invoke(record)

            delay(actionDelayMs)
        }

        return TaskResult(TaskStatus.FAILED, "Max steps ($maxSteps) exceeded", steps, stepCount)
    }

    private suspend fun executeAction(
        svc: NemoAccessibilityService,
        name: String,
        params: Map<String, Any>,
        screen: ScreenState,
    ): Boolean {
        val elems = screen.interactiveElements
        return when (name) {
            "tap" -> {
                val idx = (params["index"] as? Number)?.toInt() ?: 0
                if (idx in elems.indices) {
                    val (x, y) = elems[idx].center
                    svc.tap(x, y)
                } else false
            }
            "type_text" -> {
                val idx = (params["index"] as? Number)?.toInt() ?: 0
                val text = params["text"]?.toString() ?: ""
                if (idx in elems.indices) {
                    val (x, y) = elems[idx].center
                    svc.tap(x, y)
                    delay(300)
                    // Find the focused node and set text
                    val state = svc.getScreenState()
                    val editables = state.elements.filter { it.editable }
                    if (editables.isNotEmpty()) {
                        // Use AccessibilityService to set text directly
                        val root = svc.rootInActiveWindow
                        val nodes = root?.let { findEditableNodes(it) } ?: emptyList()
                        val typed = nodes.firstOrNull()?.let { svc.inputText(it, text) } ?: false
                        root?.recycle()
                        typed
                    } else false
                } else false
            }
            "scroll" -> {
                val direction = params["direction"]?.toString() ?: "down"
                val (w, h) = svc.getScreenSize()
                val cx = w / 2; val cy = h / 2
                val dy = h / 4
                if (direction == "down") svc.swipe(cx, cy, cx, cy - dy)
                else svc.swipe(cx, cy, cx, cy + dy)
            }
            "back" -> { svc.pressBack(); true }
            "home" -> { svc.pressHome(); true }
            else -> false
        }
    }

    private fun findEditableNodes(node: android.view.accessibility.AccessibilityNodeInfo): List<android.view.accessibility.AccessibilityNodeInfo> {
        val result = mutableListOf<android.view.accessibility.AccessibilityNodeInfo>()
        if (node.isEditable) result.add(node)
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let { result.addAll(findEditableNodes(it)) }
        }
        return result
    }

    private fun parseDecision(raw: String): JSONObject {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}') + 1
        if (start >= 0 && end > start) {
            return JSONObject(raw.substring(start, end))
        }
        throw IllegalArgumentException("No JSON found in: ${raw.take(200)}")
    }

    private fun parseParams(json: JSONObject?): Map<String, Any> {
        if (json == null) return emptyMap()
        val map = mutableMapOf<String, Any>()
        for (key in json.keys()) {
            map[key] = json.get(key)
        }
        return map
    }
}
