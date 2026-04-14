package com.nemo.agent

import com.nemo.model.EntropyRouter
import com.nemo.model.OnDeviceLLM
import com.nemo.model.PromptTemplates
import com.nemo.model.ReasoningDepth
import com.nemo.model.RoutingDecision
import com.nemo.safety.SafetyContext
import com.nemo.safety.SafetyLayer
import com.nemo.safety.Verdict
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.delay
import org.json.JSONObject

/**
 * PhoneAgent main loop on Android.
 * observe -> route -> think -> safety -> act -> learn
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
) {
    private val router = EntropyRouter()
    private val safety = if (safetyEnabled) SafetyLayer() else null
    private val planner = TaskPlanner()
    private val actionBuilder = ActionBuilder()
    private var stepCount = 0

    private val service: NemoAccessibilityService?
        get() = NemoAccessibilityService.instance

    suspend fun execute(task: String, onStep: ((StepRecord) -> Unit)? = null): TaskResult {
        stepCount = 0
        val steps = mutableListOf<StepRecord>()
        val svc = service ?: return TaskResult(TaskStatus.FAILED, "AccessibilityService not connected")

        var prevScreenHash: String? = null
        var prevActionName: String? = null
        var prevActionSuccess = false
        var activePlan: TaskPlan? = null
        var planCursor = 0

        while (stepCount < maxSteps) {
            stepCount += 1
            val startTime = System.currentTimeMillis()

            // 1. Observe
            val screen = svc.getScreenState()
            val screenHash = screen.computeHash()
            val screenContext = screen.toPromptStr()

            // Learn graph transition from previous step.
            if (prevScreenHash != null && prevActionName != null) {
                planner.recordTransition(
                    from = prevScreenHash,
                    to = screenHash,
                    actionType = prevActionName,
                    success = prevActionSuccess,
                )
            }

            // 2. Route
            val routing = if (routerEnabled) {
                router.route(screenHash)
            } else {
                RoutingDecision(
                    depth = ReasoningDepth.SYSTEM_2,
                    entropy = 1f,
                    confidence = 0f,
                    source = "router_disabled",
                )
            }

            // 3. Cache hit (System 1)
            var actionJson: String? = null
            var plannedDecision: ActionBuilder.Decision? = null
            if (routing.depth == ReasoningDepth.SYSTEM_1) {
                actionJson = router.getCachedAction(screenHash)
            }

            // 4. Optional LLM planner fallback
            if (plannerEnabled && activePlan == null) {
                activePlan = try {
                    planner.planWithLlm(
                        llm = llm,
                        task = task,
                        currentApp = screen.activity,
                        screenSummary = screenContext,
                    )
                } catch (_: Exception) {
                    null
                }
            }

            // 5. Decide next action
            if (actionJson == null) {
                val hint = planner.nextActionHint(activePlan, planCursor)
                if (
                    plannerEnabled &&
                    hint != null &&
                    routing.depth != ReasoningDepth.SYSTEM_2
                ) {
                    plannedDecision = ActionBuilder.Decision(
                        reasoning = "planner_fallback",
                        actionName = hint.actionType,
                        params = hint.params,
                    )
                    planCursor += 1
                } else {
                    val history = buildRecentHistory(steps)
                    val detail = llm.decideActionDetailed(
                        systemPrompt = PromptTemplates.resolveSystemPrompt(promptVersion),
                        screenContext = screenContext,
                        task = task,
                        history = history,
                        promptVersion = promptVersion,
                    )
                    if (detail.entropy != null) {
                        router.observeEntropy(screenHash, detail.entropy)
                    }
                    actionJson = detail.rawText
                }
            }

            // 6. Parse decision
            val decision = plannedDecision ?: try {
                actionBuilder.parseDecision(actionJson ?: "")
            } catch (e: Exception) {
                val record = StepRecord(
                    step = stepCount,
                    action = "error",
                    success = false,
                    error = e.message ?: "parse error",
                )
                steps += record
                onStep?.invoke(record)
                prevScreenHash = screenHash
                prevActionName = "error"
                prevActionSuccess = false
                continue
            }

            val actionName = decision.actionName
            val reasoning = decision.reasoning
            val params = decision.params

            // Done check
            if (actionName == "done") {
                val record = StepRecord(
                    step = stepCount,
                    action = "done",
                    params = params,
                    reasoning = reasoning,
                    success = true,
                )
                steps += record
                onStep?.invoke(record)
                return TaskResult(
                    status = TaskStatus.COMPLETED,
                    summary = params["summary"]?.toString() ?: reasoning,
                    steps = steps,
                    totalSteps = stepCount,
                )
            }

            // 7. Safety
            if (safety != null) {
                val ctx = SafetyContext(
                    screenText = screenContext,
                    stepCount = stepCount,
                    maxSteps = maxSteps,
                    actionName = actionName,
                )
                val result = safety.check(ctx)
                if (result.verdict == Verdict.BLOCK) {
                    val record = StepRecord(
                        step = stepCount,
                        action = actionName,
                        params = params,
                        reasoning = reasoning,
                        success = false,
                        error = "Safety: ${result.message}",
                    )
                    steps += record
                    onStep?.invoke(record)
                    return TaskResult(
                        status = TaskStatus.BLOCKED,
                        summary = result.message,
                        steps = steps,
                        totalSteps = stepCount,
                    )
                }
            }

            // 8. Build and execute
            val builtAction = actionBuilder.buildAction(actionName, params, screen)
            if (builtAction == null) {
                val record = StepRecord(
                    step = stepCount,
                    action = actionName,
                    params = params,
                    reasoning = reasoning,
                    success = false,
                    error = "Unknown or invalid action: $actionName",
                )
                steps += record
                onStep?.invoke(record)
                prevScreenHash = screenHash
                prevActionName = actionName
                prevActionSuccess = false
                continue
            }

            val success = actionBuilder.executeAction(svc, builtAction)
            val duration = System.currentTimeMillis() - startTime

            if (success) {
                val cachePayload = actionJson ?: decisionToJson(decision)
                router.cacheAction(screenHash, cachePayload)
            }

            val record = StepRecord(
                step = stepCount,
                action = actionName,
                params = params,
                reasoning = reasoning,
                success = success,
                durationMs = duration,
            )
            steps += record
            onStep?.invoke(record)

            prevScreenHash = screenHash
            prevActionName = actionName
            prevActionSuccess = success

            delay(actionDelayMs)
        }

        return TaskResult(
            status = TaskStatus.FAILED,
            summary = "Max steps ($maxSteps) exceeded",
            steps = steps,
            totalSteps = stepCount,
        )
    }

    private fun buildRecentHistory(steps: List<StepRecord>): List<PromptTemplates.HistoryItem> {
        if (contextWindowSteps <= 0 || steps.isEmpty()) return emptyList()
        return steps.takeLast(contextWindowSteps).map {
            PromptTemplates.HistoryItem(
                step = it.step,
                action = it.action,
                success = it.success,
                error = it.error,
            )
        }
    }

    private fun decisionToJson(decision: ActionBuilder.Decision): String {
        val paramsJson = JSONObject()
        for ((k, v) in decision.params) {
            paramsJson.put(k, v)
        }
        return JSONObject()
            .put("reasoning", decision.reasoning)
            .put("action", decision.actionName)
            .put("params", paramsJson)
            .toString()
    }
}
