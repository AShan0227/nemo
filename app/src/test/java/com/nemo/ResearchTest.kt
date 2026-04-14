package com.nemo

import android.graphics.Rect
import com.nemo.agent.ActionBuilder
import com.nemo.agent.PlanStep
import com.nemo.agent.TaskPlan
import com.nemo.agent.TaskPlanner
import com.nemo.model.EntropyRouter
import com.nemo.model.PromptTemplates
import com.nemo.model.ReasoningDepth
import com.nemo.model.RoutingBenchmarkCase
import com.nemo.model.compareRoutingAccuracy
import com.nemo.research.BoltzmannExplorer
import com.nemo.research.CircadianModel
import com.nemo.research.HomeostasisRegulator
import com.nemo.research.ImmuneSystem
import com.nemo.research.InertiaController
import com.nemo.research.IntentTracker
import com.nemo.research.PerformanceSnapshot
import com.nemo.research.PhaseDetector
import com.nemo.screen.SignalCandidate
import com.nemo.screen.ScreenState
import com.nemo.screen.UIElement
import com.nemo.screen.fuseScreenSources
import java.util.Calendar
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ResearchTest {

    @Test
    fun entropyRouter_entropy_singleOption_isZero() {
        val router = EntropyRouter()
        assertEquals(0f, router.computeEntropy(listOf(1f)), 1e-6f)
    }

    @Test
    fun entropyRouter_entropy_uniformTwo_isHigh() {
        val router = EntropyRouter()
        val e = router.computeEntropy(listOf(0.5f, 0.5f))
        assertTrue(e > 0.99f)
    }

    @Test
    fun entropyRouter_cache_enablesSystem1() {
        val router = EntropyRouter()
        router.cacheAction("s", "{}")
        val decision = router.route("s", entropy = 0.1f)
        assertEquals(ReasoningDepth.SYSTEM_1, decision.depth)
    }

    @Test
    fun entropyRouter_observedEntropy_usedOnNextRoute() {
        val router = EntropyRouter()
        router.observeEntropy("s2", 0.2f)
        val decision = router.route("s2")
        assertEquals("observed_entropy", decision.source)
        assertEquals(ReasoningDepth.SYSTEM_1_5, decision.depth)
    }

    @Test
    fun routingBenchmark_reportHasDelta() {
        val report = compareRoutingAccuracy(
            cases = listOf(
                RoutingBenchmarkCase(
                    screenHash = "a",
                    expectedDepth = ReasoningDepth.SYSTEM_2,
                    actionProbs = listOf(0.5f, 0.5f),
                    realEntropy = 0.9f,
                ),
            ),
        )
        assertEquals(1, report.sampleSize)
        assertTrue(report.delta <= 1f)
    }

    @Test
    fun taskPlanner_graphPlan_singleEdge() {
        val planner = TaskPlanner()
        planner.recordTransition("a", "b", "tap", success = true)
        val plan = planner.planFromGraph("a", "b")
        assertNotNull(plan)
        assertEquals(1, plan!!.steps.size)
        assertEquals("tap", plan.steps.first().actionType)
    }

    @Test
    fun taskPlanner_nextActionHint_bounds() {
        val planner = TaskPlanner()
        val plan = TaskPlan(
            task = "t",
            steps = listOf(
                PlanStep(1, "tap", description = "d", expectedState = "s", estimatedCost = 1f),
            ),
            totalCost = 1f,
            confidence = 0.5f,
        )
        assertNotNull(planner.nextActionHint(plan, 0))
        assertNull(planner.nextActionHint(plan, 1))
    }

    @Test
    fun actionBuilder_parseDecision_json() {
        val builder = ActionBuilder()
        val d = builder.parseDecision("{" +
            "\"reasoning\":\"ok\"," +
            "\"action\":\"tap\"," +
            "\"params\":{\"index\":1}}")
        assertEquals("tap", d.actionName)
        assertEquals(1, (d.params["index"] as Number).toInt())
    }

    @Test
    fun actionBuilder_buildTap_validIndex() {
        val builder = ActionBuilder()
        val screen = ScreenState(
            activity = "app",
            packageName = "pkg",
            elements = listOf(
                UIElement(index = 0, className = "Button", bounds = Rect(0, 0, 100, 100), clickable = true),
            ),
        )
        val action = builder.buildAction("tap", mapOf("index" to 0), screen)
        assertNotNull(action)
    }

    @Test
    fun actionBuilder_buildTap_invalidIndex_returnsNull() {
        val builder = ActionBuilder()
        val screen = ScreenState(elements = emptyList())
        val action = builder.buildAction("tap", mapOf("index" to 5), screen)
        assertNull(action)
    }

    @Test
    fun promptTemplates_baseline_hasNoFewShot() {
        val messages = PromptTemplates.buildDecisionMessages(
            task = "task",
            screenContext = "screen",
            promptVersion = "baseline_v1",
        )
        assertEquals(2, messages.size)
    }

    @Test
    fun promptTemplates_reflect_hasFewShot() {
        val messages = PromptTemplates.buildDecisionMessages(
            task = "task",
            screenContext = "screen",
            promptVersion = "reflect_fewshot_v1",
        )
        assertTrue(messages.size > 2)
    }

    @Test
    fun promptTemplates_renderPrompt_containsRoles() {
        val text = PromptTemplates.renderMessagesAsPrompt(
            listOf(PromptTemplates.Message("system", "s"), PromptTemplates.Message("user", "u")),
        )
        assertTrue(text.contains("<system>"))
        assertTrue(text.contains("<user>"))
    }

    @Test
    fun homeostasis_adjustments_triggerOnLowSuccess() {
        val h = HomeostasisRegulator()
        h.updateMetrics(successRate = 0.3f)
        val adj = h.getAdjustments()
        assertTrue(adj.any { it.name == "verify_actions" })
    }

    @Test
    fun homeostasis_adjustments_triggerOnHighError() {
        val h = HomeostasisRegulator()
        h.updateMetrics(errorRate = 0.7f)
        val adj = h.getAdjustments()
        assertTrue(adj.any { it.name == "increase_delay" })
    }

    @Test
    fun circadian_recordAndPredict() {
        val c = CircadianModel()
        c.recordActivity("com.test.app", "tap")
        val predicted = c.predictApps(topK = 1)
        assertTrue(predicted.isEmpty() || predicted.first() == "com.test.app")
    }

    @Test
    fun circadian_modifier_nightHasHigherDelay() {
        val c = CircadianModel()
        val oneAm = Calendar.getInstance().apply {
            set(2026, Calendar.JANUARY, 1, 1, 0, 0)
            set(Calendar.MILLISECOND, 0)
        }.timeInMillis
        val mods = c.getBehaviorModifier(oneAm)
        assertTrue(mods.actionDelayMultiplier > 1f)
    }

    @Test
    fun immune_trainAndCheck_runs() {
        val immune = ImmuneSystem(detectorCount = 10)
        repeat(15) {
            immune.addSelfSample(
                com.nemo.research.ScreenFeatures(10, 3, 20, false, true, false, 4, 2),
            )
        }
        val trained = immune.train()
        assertTrue(trained > 0)
        val result = immune.check(
            com.nemo.research.ScreenFeatures(12, 4, 30, true, true, false, 5, 3),
        )
        assertNotNull(result)
    }

    @Test
    fun inertia_followPlan_whenLowConfidence() {
        val inertia = InertiaController()
        inertia.setPlan(listOf("tap", "scroll"))
        val d = inertia.shouldFollowPlan("tap", "scroll", newConfidence = 0.1f)
        assertTrue(d.usePlanned)
    }

    @Test
    fun inertia_override_whenHighConfidence() {
        val inertia = InertiaController(baseInertia = 0.2f)
        inertia.setPlan(listOf("tap"))
        val d = inertia.shouldFollowPlan("tap", "scroll", newConfidence = 0.9f)
        assertFalse(d.usePlanned)
    }

    @Test
    fun explorer_selectAction_returnsKnownAction() {
        val explorer = BoltzmannExplorer()
        val (action, prob) = explorer.selectAction("s", listOf("tap", "scroll"))
        assertTrue(action == "tap" || action == "scroll")
        assertTrue(prob > 0f)
    }

    @Test
    fun intentTracker_updatesState() {
        val tracker = IntentTracker(listOf("open settings", "send message"))
        tracker.updateEvidence("open settings now")
        val state = tracker.getState()
        assertNotNull(state)
        assertTrue(state.entropy in 0f..1f)
    }

    @Test
    fun phaseDetector_detectsImprovement() {
        val detector = PhaseDetector(windowSize = 3, baselineSize = 12, sensitivity = 1f)
        repeat(9) {
            detector.record(
                PerformanceSnapshot(
                    timestamp = it.toLong(),
                    successRate = 0.2f,
                    avgSteps = 20f,
                    graphNodes = 10,
                    graphEdges = 10,
                ),
            )
        }
        val transition = detector.record(
            PerformanceSnapshot(
                timestamp = 99,
                successRate = 0.95f,
                avgSteps = 5f,
                graphNodes = 80,
                graphEdges = 80,
            ),
        )
        assertNotNull(transition)
    }

    @Test
    fun dempster_fusion_prefers_sharedEvidence() {
        val ui = listOf(SignalCandidate(label = "Wi-Fi", confidence = 0.9, source = "ui"))
        val ocr = listOf(SignalCandidate(label = "Wi-Fi", confidence = 0.8, source = "ocr"))
        val result = fuseScreenSources(uiCandidates = ui, ocrCandidates = ocr)

        assertEquals("wi-fi", result.bestLabel)
        assertTrue(result.conflict < 0.5)
    }
}
