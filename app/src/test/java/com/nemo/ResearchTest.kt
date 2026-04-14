package com.nemo

import com.nemo.model.EntropyRouter
import com.nemo.model.ReasoningDepth
import com.nemo.research.*
import com.nemo.safety.SafetyContext
import com.nemo.safety.SafetyLayer
import com.nemo.safety.Verdict
import org.junit.Assert.*
import org.junit.Test

class ResearchTest {

    // --- EntropyRouter ---

    @Test
    fun `entropy of single option is zero`() {
        val router = EntropyRouter()
        assertEquals(0f, router.computeEntropy(listOf(1f)), 0.01f)
    }

    @Test
    fun `entropy of uniform distribution is max`() {
        val router = EntropyRouter()
        val entropy = router.computeEntropy(listOf(0.5f, 0.5f))
        assertEquals(1f, entropy, 0.01f)
    }

    @Test
    fun `system1 cache hit`() {
        val router = EntropyRouter()
        router.cacheAction("screen1", """{"action":"tap"}""")
        val decision = router.route("screen1", listOf(0.99f, 0.01f))
        assertEquals(ReasoningDepth.SYSTEM_1, decision.depth)
    }

    // --- ImmuneSystem ---

    @Test
    fun `immune not trained returns normal`() {
        val immune = ImmuneSystem()
        val features = ScreenFeatures(20, 8, 200, true, true, true, 5, 3)
        assertFalse(immune.check(features).isAnomaly)
    }

    @Test
    fun `immune trains with enough samples`() {
        val immune = ImmuneSystem(detectorCount = 10)
        repeat(20) {
            immune.addSelfSample(ScreenFeatures(20, 8, 200, true, true, true, 5, 3))
        }
        val count = immune.train()
        assertTrue(count > 0)
        assertTrue(immune.trained)
    }

    // --- Homeostasis ---

    @Test
    fun `no adjustment when healthy`() {
        val reg = HomeostasisRegulator()
        reg.updateMetrics(successRate = 0.95f, latencyMs = 1500f, errorRate = 0.02f)
        assertTrue(reg.getAdjustments().isEmpty())
    }

    @Test
    fun `increase delay on high errors`() {
        val reg = HomeostasisRegulator()
        reg.updateMetrics(errorRate = 0.3f)
        val names = reg.getAdjustments().map { it.name }
        assertTrue("increase_delay" in names)
    }

    // --- InertiaController ---

    @Test
    fun `no plan means no inertia`() {
        val ctrl = InertiaController()
        assertFalse(ctrl.hasPlan)
        assertEquals(0f, ctrl.computeInertia(), 0.01f)
    }

    @Test
    fun `plan agrees keeps plan`() {
        val ctrl = InertiaController()
        ctrl.setPlan(listOf("tap", "scroll", "done"))
        val d = ctrl.shouldFollowPlan("tap", "tap")
        assertTrue(d.usePlanned)
    }

    @Test
    fun `high confidence overrides plan`() {
        val ctrl = InertiaController(baseInertia = 0.3f)
        ctrl.setPlan(listOf("tap", "done"))
        val d = ctrl.shouldFollowPlan("tap", "scroll", 0.9f)
        assertFalse(d.usePlanned)
    }

    // --- BoltzmannExplorer ---

    @Test
    fun `explorer updates q values`() {
        val exp = BoltzmannExplorer()
        exp.updateValue("s1", "tap", 5f)
        assertEquals(1, exp.knownStates)
    }

    @Test
    fun `annealing reduces temperature`() {
        val exp = BoltzmannExplorer(initialTemperature = 1f, annealingRate = 0.5f)
        exp.anneal()
        assertEquals(0.5f, exp.temperature, 0.01f)
    }

    // --- PhaseDetector ---

    @Test
    fun `no transition with little data`() {
        val detector = PhaseDetector(windowSize = 5)
        repeat(3) {
            assertNull(detector.record(PerformanceSnapshot(System.currentTimeMillis(), 0.5f, 10f, 5, 5)))
        }
    }

    @Test
    fun `detects success rate jump`() {
        val detector = PhaseDetector(windowSize = 5, sensitivity = 1.5f)
        repeat(30) { detector.record(PerformanceSnapshot(it.toLong(), 0.5f, 10f, 5, 5)) }
        var found = false
        repeat(10) {
            val t = detector.record(PerformanceSnapshot((30 + it).toLong(), 0.95f, 10f, 5, 5))
            if (t != null) found = true
        }
        assertTrue(found)
    }

    // --- CircadianModel ---

    @Test
    fun `records activity`() {
        val model = CircadianModel()
        model.recordActivity("com.tencent.mm", "tap")
        assertEquals(1, model.totalActivities)
    }

    // --- IntentTracker ---

    @Test
    fun `collapses with evidence`() {
        val tracker = IntentTracker(listOf("open system settings", "open app settings"), collapseThreshold = 0.7f)
        repeat(5) { tracker.updateEvidence("Wi-Fi Bluetooth System settings panel") }
        assertTrue(tracker.isCollapsed)
        assertTrue("system" in tracker.getBestIntent())
    }

    // --- StrategyEvolver ---

    @Test
    fun `evolution cycle runs`() {
        val evolver = StrategyEvolver(populationSize = 5)
        val g = evolver.getStrategy()
        evolver.reportFitness(g, true, 8, 5000f)
        evolver.evolve()
        assertEquals(1, evolver.generation)
    }

    // --- SafetyLayer ---

    @Test
    fun `allows normal action`() {
        val safety = SafetyLayer()
        val result = safety.check(SafetyContext("Home screen", 1, 30))
        assertEquals(Verdict.ALLOW, result.verdict)
    }

    @Test
    fun `blocks financial`() {
        val safety = SafetyLayer()
        val result = safety.check(SafetyContext("Confirm purchase \$99", 1, 30))
        assertEquals(Verdict.BLOCK, result.verdict)
    }

    @Test
    fun `blocks chinese financial`() {
        val safety = SafetyLayer()
        val result = safety.check(SafetyContext("确认支付 99元", 1, 30))
        assertEquals(Verdict.BLOCK, result.verdict)
    }
}
