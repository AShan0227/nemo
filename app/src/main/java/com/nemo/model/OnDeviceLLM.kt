package com.nemo.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * On-device LLM inference via LiteRT-LM.
 *
 * Supports .litertlm (Gemma 4 E2B) and .task (Gemma 3 1B) formats.
 * Uses Google's LiteRT-LM Engine API: Engine → Conversation → sendMessage().
 */
data class LLMRuntimeConfig(
    val maxOutputTokens: Int = 256,
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val temperature: Float = 0.15f,
    val randomSeed: Int = 7,
    val promptTokenLimit: Int = 4096,
)

data class PromptCompressionStats(
    val originalPromptTokens: Int,
    val finalPromptTokens: Int,
    val droppedHistoryItems: Int,
    val screenContextTrimmed: Boolean,
)

data class LLMDecisionResult(
    val rawText: String,
    val entropy: Float? = null,
    val entropySource: String = "none",
    val latencyMs: Long = 0L,
    val promptTokens: Int = 0,
    val compressionStats: PromptCompressionStats? = null,
)

data class GemmaBenchmarkResult(
    val modelLoadMs: Long,
    val warmupMs: Long,
    val avgLatencyMs: Float,
    val p95LatencyMs: Long,
    val minLatencyMs: Long,
    val maxLatencyMs: Long,
    val validJsonRate: Float,
    val logprobsAvailable: Boolean,
    val promptTokens: Int,
    val meetsLatencyTarget: Boolean,
    val iterations: Int,
)

class OnDeviceLLM(
    private val context: Context,
    private val modelPath: String,
    private var runtimeConfig: LLMRuntimeConfig = LLMRuntimeConfig(),
) {
    // LiteRT-LM Engine (loaded via reflection for compile safety)
    private var engine: Any? = null
    private var loaded = false

    val isLoaded: Boolean get() = loaded

    fun updateRuntimeConfig(config: LLMRuntimeConfig) { runtimeConfig = config }

    suspend fun load() = withContext(Dispatchers.IO) {
        if (loaded && engine != null) return@withContext
        try {
            // LiteRT-LM API: EngineConfig → Engine → initialize()
            val engineConfigClass = Class.forName("com.google.ai.edge.litertlm.EngineConfig")
            val engineClass = Class.forName("com.google.ai.edge.litertlm.Engine")

            // Create EngineConfig with model path
            val configConstructor = engineConfigClass.constructors.first()
            val engineConfig = configConstructor.newInstance(modelPath)

            // Create Engine and initialize
            val engineConstructor = engineClass.constructors.first()
            engine = engineConstructor.newInstance(engineConfig)

            val initMethod = engineClass.getMethod("initialize")
            initMethod.invoke(engine)

            loaded = true
        } catch (e: Exception) {
            throw RuntimeException(
                "Failed to load model at: $modelPath\n" +
                "Error: ${e.cause?.message ?: e.message}\n" +
                "Ensure the model file is a valid .litertlm or .task file.",
                e,
            )
        }
    }

    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        val eng = engine ?: throw IllegalStateException("Model not loaded. Call load() first.")
        try {
            val engineClass = eng.javaClass
            val createConvMethod = engineClass.getMethod("createConversation")
            val conversation = createConvMethod.invoke(eng)

            val convClass = conversation.javaClass
            val sendMethod = convClass.getMethod("sendMessage", String::class.java)
            val response = sendMethod.invoke(conversation, prompt) as? String ?: ""

            // Close conversation
            try { convClass.getMethod("close").invoke(conversation) } catch (_: Exception) {}

            response
        } catch (e: Exception) {
            throw RuntimeException("Inference failed: ${e.cause?.message ?: e.message}", e)
        }
    }

    suspend fun generateDetailed(prompt: String): LLMDecisionResult = withContext(Dispatchers.IO) {
        val startTime = System.nanoTime()
        val text = generate(prompt)
        val latencyMs = (System.nanoTime() - startTime) / 1_000_000
        LLMDecisionResult(
            rawText = text,
            latencyMs = latencyMs,
            promptTokens = prompt.length / 4,
        )
    }

    suspend fun decideAction(
        systemPrompt: String,
        screenContext: String,
        task: String,
        history: List<PromptTemplates.HistoryItem> = emptyList(),
        promptVersion: String = "reflect_fewshot_v1",
    ): String = decideActionDetailed(systemPrompt, screenContext, task, history, promptVersion).rawText

    suspend fun decideActionDetailed(
        systemPrompt: String,
        screenContext: String,
        task: String,
        history: List<PromptTemplates.HistoryItem> = emptyList(),
        promptVersion: String = "reflect_fewshot_v1",
    ): LLMDecisionResult = withContext(Dispatchers.IO) {
        val messages = PromptTemplates.buildDecisionMessages(
            task = task, screenContext = screenContext,
            history = history, promptVersion = promptVersion,
        )
        val fullPrompt = PromptTemplates.renderMessagesAsPrompt(messages)

        // Trim if over budget
        val trimmed = if (fullPrompt.length > runtimeConfig.promptTokenLimit * 4) {
            fullPrompt.take(runtimeConfig.promptTokenLimit * 4)
        } else fullPrompt

        val detail = generateDetailed(trimmed)
        val normalized = normalizeDecisionJson(detail.rawText)
        detail.copy(rawText = normalized)
    }

    suspend fun runDecisionBenchmark(
        iterations: Int = 3,
        warmupRounds: Int = 1,
        task: String = "Open Settings",
        screenContext: String = "Screen: Home\nInteractive elements (3):\n  [0] \"Settings\" Button [clickable]\n  [1] \"Camera\" Button [clickable]\n  [2] \"Phone\" Button [clickable]",
    ): GemmaBenchmarkResult = withContext(Dispatchers.IO) {
        val loadStart = System.nanoTime()
        if (!loaded) load()
        val modelLoadMs = (System.nanoTime() - loadStart) / 1_000_000

        var warmupMs = 0L
        repeat(warmupRounds) {
            val d = decideActionDetailed("", screenContext, task, promptVersion = "baseline_v1")
            warmupMs += d.latencyMs
        }

        val latencies = mutableListOf<Long>()
        var validJson = 0
        repeat(iterations) {
            val d = decideActionDetailed("", screenContext, task, promptVersion = "baseline_v1")
            latencies += d.latencyMs
            if (isValidDecisionJson(d.rawText)) validJson++
        }

        val sorted = latencies.sorted()
        GemmaBenchmarkResult(
            modelLoadMs = modelLoadMs, warmupMs = warmupMs,
            avgLatencyMs = latencies.average().toFloat(),
            p95LatencyMs = sorted[(sorted.size * 0.95).toInt().coerceIn(0, sorted.size - 1)],
            minLatencyMs = sorted.first(), maxLatencyMs = sorted.last(),
            validJsonRate = validJson.toFloat() / iterations,
            logprobsAvailable = false, promptTokens = 0,
            meetsLatencyTarget = latencies.average() < 500,
            iterations = iterations,
        )
    }

    fun close() {
        engine?.let {
            try { it.javaClass.getMethod("close").invoke(it) } catch (_: Exception) {}
        }
        engine = null
        loaded = false
    }

    private fun normalizeDecisionJson(raw: String): String {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}') + 1
        if (start >= 0 && end > start) {
            try {
                val obj = org.json.JSONObject(raw.substring(start, end))
                if (!obj.has("action")) {
                    obj.put("action", "wait")
                    obj.put("params", org.json.JSONObject().put("ms", 1000))
                    obj.put("reasoning", "fallback_no_action")
                }
                return obj.toString()
            } catch (_: Exception) {}
        }
        return """{"reasoning":"model_output_unparseable","action":"wait","params":{"ms":1000}}"""
    }

    private fun isValidDecisionJson(text: String): Boolean {
        return try {
            val s = text.indexOf('{'); val e = text.lastIndexOf('}') + 1
            if (s < 0 || e <= s) return false
            org.json.JSONObject(text.substring(s, e)).has("action")
        } catch (_: Exception) { false }
    }
}
