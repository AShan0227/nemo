package com.nemo.agent

import com.nemo.knowledge.GraphPersistence
import com.nemo.knowledge.ScreenGraph
import com.nemo.model.EntropyRouter
import com.nemo.model.OnDeviceLLM
import com.nemo.model.PromptTemplates
import com.nemo.model.ReasoningDepth
import com.nemo.model.RoutingDecision
import com.nemo.research.*
import com.nemo.safety.SafetyContext
import com.nemo.safety.SafetyLayer
import com.nemo.safety.Verdict
import com.nemo.screen.ScreenReader
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.delay
import org.json.JSONObject

/**
 * PhoneAgent — full Android agent loop with all research mechanisms.
 *
 * 1. Observe (AccessibilityService)
 * 2. Immune check (anomaly detection)
 * 3. Homeostasis + Circadian (behavior adjustment)
 * 4. Route (entropy-based compute depth)
 * 5. Plan (A* + LLM fallback)
 * 6. Think (on-device Gemma / cache)
 * 7. Inertia (plan stability)
 * 8. Safety (invariant filter)
 * 9. Act (gesture dispatch)
 * 10. Explorer (Q-value update)
 * 11. Learn (graph transition + cache)
 * 12. Phase detect (capability jumps)
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
    private val routerEnabled: Boolean = true,
    private val plannerEnabled: Boolean = true,
    private val contextWindowSteps: Int = 4,
    private val promptVersion: String = "reflect_fewshot_v1",
    // Research mechanism toggles
    private val homeostasisEnabled: Boolean = true,
    private val immuneEnabled: Boolean = false,
    private val inertiaEnabled: Boolean = true,
    private val explorerEnabled: Boolean = false,
    private val phaseDetectorEnabled: Boolean = true,
    private val circadianEnabled: Boolean = true,
) {
    private val router = EntropyRouter()
    private val safety = if (safetyEnabled) SafetyLayer() else null
    private val planner = TaskPlanner()
    private val actionBuilder = ActionBuilder()
    private var graph = ScreenGraph()
    private var graphPersistence: GraphPersistence? = null

    // Research mechanisms
    private val homeostasis = if (homeostasisEnabled) HomeostasisRegulator() else null
    private val immune = if (immuneEnabled) ImmuneSystem() else null
    private val inertia = if (inertiaEnabled) InertiaController() else null
    private val explorer = if (explorerEnabled) BoltzmannExplorer() else null
    private val phaseDetector = if (phaseDetectorEnabled) PhaseDetector() else null
    private val circadian = if (circadianEnabled) CircadianModel() else null

    private var stepCount = 0
    private val successWindow = mutableListOf<Boolean>()

    private val service: NemoAccessibilityService?
        get() = NemoAccessibilityService.instance

    /**
     * Initialize Room-based graph persistence. Call once with Android Context.
     * Loads previously learned navigation paths from SQLite.
     */
    suspend fun initPersistence(context: android.content.Context) {
        val persistence = GraphPersistence(context)
        graphPersistence = persistence
        graph = persistence.loadGraph()
    }

    suspend fun execute(task: String, onStep: ((StepRecord) -> Unit)? = null): TaskResult {
        stepCount = 0
        successWindow.clear()
        var currentTask = task
        val steps = mutableListOf<StepRecord>()
        val svc = service ?: return TaskResult(TaskStatus.FAILED, "AccessibilityService not connected")

        var prevScreenHash: String? = null
        var activePlan: TaskPlan? = null
        var planCursor = 0

        while (stepCount < maxSteps) {
            stepCount += 1
            val startTime = System.currentTimeMillis()

            // 1. Observe
            val screen = svc.getScreenState()
            val screenHash = screen.computeHash()
            val screenContext = screen.toPromptStr()

            // 2. Immune check — detect anomalous screens
            if (immune != null) {
                val features = ScreenFeatures(
                    elementCount = screen.elements.size,
                    interactiveCount = screen.interactiveElements.size,
                    textLength = screen.elements.sumOf { it.displayText.length },
                    hasInput = screen.elements.any { it.editable },
                    hasButton = screen.elements.any { it.clickable && "Button" in it.className },
                    hasScrollable = screen.elements.any { it.scrollable },
                    uniqueClasses = screen.elements.map { it.className }.toSet().size,
                    depthEstimate = 0,
                )
                if (immune.trained) {
                    val anomaly = immune.check(features)
                    if (anomaly.isAnomaly) {
                        svc.pressBack()
                        delay(500)
                        continue
                    }
                } else {
                    immune.addSelfSample(features)
                    if (immune.selfSetSize >= 20) immune.train()
                }
            }

            // 3. Homeostasis + Circadian
            if (homeostasis != null) {
                for (adj in homeostasis.getAdjustments()) {
                    if (adj.name == "increase_delay" && adj.value > 0.1f) {
                        // Could adjust delay dynamically — skipped for immutable param
                    }
                }
            }

            // Learn graph transition from previous step
            if (prevScreenHash != null) {
                val prevAction = steps.lastOrNull()?.action ?: "unknown"
                val prevSuccess = steps.lastOrNull()?.success ?: false
                planner.recordTransition(prevScreenHash, screenHash, prevAction, prevSuccess)
                graph.recordTransition(prevScreenHash, screenHash, prevAction, prevSuccess)
            }

            // 4. Route
            val routing = if (routerEnabled) {
                router.route(screenHash)
            } else {
                RoutingDecision(ReasoningDepth.SYSTEM_2, 1f, 0f, "router_disabled")
            }

            // 5. Cache hit (System 1)
            var actionJson: String? = null
            var plannedDecision: ActionBuilder.Decision? = null
            if (routing.depth == ReasoningDepth.SYSTEM_1) {
                actionJson = router.getCachedAction(screenHash)
            }

            // 5b. Planner LLM fallback
            if (plannerEnabled && activePlan == null) {
                activePlan = try {
                    planner.planWithLlm(llm, currentTask, screen.activity, screenContext)
                } catch (_: Exception) { null }
            }

            // 6. Think
            if (actionJson == null && plannedDecision == null) {
                val hint = planner.nextActionHint(activePlan, planCursor)
                if (plannerEnabled && hint != null && routing.depth != ReasoningDepth.SYSTEM_2) {
                    plannedDecision = ActionBuilder.Decision("planner_fallback", hint.actionType, hint.params)
                    planCursor++
                } else {
                    val history = buildRecentHistory(steps)
                    val detail = llm.decideActionDetailed(
                        systemPrompt = PromptTemplates.resolveSystemPrompt(promptVersion),
                        screenContext = screenContext, task = currentTask,
                        history = history, promptVersion = promptVersion,
                    )
                    if (detail.entropy != null) router.observeEntropy(screenHash, detail.entropy)
                    actionJson = detail.rawText
                }
            }

            // 7. Parse
            val decision = plannedDecision ?: try {
                actionBuilder.parseDecision(actionJson ?: "")
            } catch (e: Exception) {
                val record = StepRecord(stepCount, "error", error = e.message ?: "parse error")
                steps += record; onStep?.invoke(record)
                successWindow.add(false)
                prevScreenHash = screenHash
                continue
            }

            val actionName = decision.actionName
            val reasoning = decision.reasoning
            val params = decision.params

            // Done check
            if (actionName == "done") {
                val record = StepRecord(stepCount, "done", params, reasoning, true)
                steps += record; onStep?.invoke(record)
                inertia?.clearPlan()
                val result = TaskResult(TaskStatus.COMPLETED, params["summary"]?.toString() ?: reasoning, steps, stepCount)
                onTaskComplete(result)
                return result
            }

            // 7b. Inertia — plan stability check
            var resolvedAction = actionName
            if (inertia != null && inertia.hasPlan) {
                val planned = inertia.getNextPlannedAction()
                if (planned != null && planned != actionName) {
                    val decision2 = inertia.shouldFollowPlan(planned, actionName, routing.confidence)
                    if (decision2.usePlanned) resolvedAction = planned
                }
            }

            // 8. Safety
            if (safety != null) {
                val ctx = SafetyContext(screenContext, stepCount, maxSteps, resolvedAction)
                val result = safety.check(ctx)
                if (result.verdict == Verdict.BLOCK) {
                    val record = StepRecord(stepCount, resolvedAction, params, reasoning, false, error = "Safety: ${result.message}")
                    steps += record; onStep?.invoke(record)
                    return TaskResult(TaskStatus.BLOCKED, result.message, steps, stepCount)
                }
            }

            // 9. Build + Execute
            val builtAction = actionBuilder.buildAction(resolvedAction, params, screen)
            if (builtAction == null) {
                val record = StepRecord(stepCount, resolvedAction, params, reasoning, false, error = "Invalid action: $resolvedAction")
                steps += record; onStep?.invoke(record)
                successWindow.add(false)
                prevScreenHash = screenHash
                continue
            }

            val success = actionBuilder.executeAction(svc, builtAction)
            val duration = System.currentTimeMillis() - startTime
            successWindow.add(success)

            if (success) router.cacheAction(screenHash, actionJson ?: decisionToJson(decision))

            val record = StepRecord(stepCount, resolvedAction, params, reasoning, success, duration)
            steps += record; onStep?.invoke(record)

            // 10. Explorer — update Q-values
            if (explorer != null) {
                explorer.updateValue(screenHash, resolvedAction, if (success) 1f else -0.5f)
                explorer.anneal()
            }

            // 11. Inertia advance + Circadian record
            if (inertia != null && success) inertia.advance()
            if (circadian != null && screen.packageName.isNotBlank()) {
                circadian.recordActivity(screen.packageName, resolvedAction, duration.toFloat())
            }

            // 12. Homeostasis metrics update
            if (homeostasis != null) {
                val recent = successWindow.takeLast(20)
                val sr = recent.count { it }.toFloat() / recent.size
                homeostasis.updateMetrics(successRate = sr, latencyMs = duration.toFloat(), errorRate = 1f - sr)
            }

            // Periodic pheromone evaporation
            if (stepCount % 10 == 0) graph.evaporatePheromones()

            prevScreenHash = screenHash
            delay(actionDelayMs)
        }

        val result = TaskResult(TaskStatus.FAILED, "Max steps ($maxSteps) exceeded", steps, stepCount)
        onTaskComplete(result)
        return result
    }

    private fun onTaskComplete(result: TaskResult) {
        if (phaseDetector != null) {
            val recent = successWindow.takeLast(20)
            val sr = if (recent.isNotEmpty()) recent.count { it }.toFloat() / recent.size else 0f
            phaseDetector.record(PerformanceSnapshot(
                timestamp = System.currentTimeMillis(),
                successRate = sr,
                avgSteps = result.totalSteps.toFloat(),
                graphNodes = graph.nodeCount,
                graphEdges = graph.edgeCount,
            ))
        }
        // Persist learned graph to SQLite
        graphPersistence?.let { persistence ->
            kotlinx.coroutines.runBlocking {
                try { persistence.saveGraph(graph) } catch (_: Exception) {}
            }
        }
    }

    private fun buildRecentHistory(steps: List<StepRecord>): List<PromptTemplates.HistoryItem> {
        if (contextWindowSteps <= 0 || steps.isEmpty()) return emptyList()
        return steps.takeLast(contextWindowSteps).map {
            PromptTemplates.HistoryItem(it.step, it.action, it.success, it.error)
        }
    }

    private fun decisionToJson(decision: ActionBuilder.Decision): String {
        val paramsJson = JSONObject()
        for ((k, v) in decision.params) paramsJson.put(k, v)
        return JSONObject().put("reasoning", decision.reasoning)
            .put("action", decision.actionName).put("params", paramsJson).toString()
    }
}
