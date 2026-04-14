package com.nemo.model

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * On-device LLM inference via MediaPipe + Gemma.
 */
data class LLMRuntimeConfig(
    val maxOutputTokens: Int = 256,   // Gemma 3n supports longer outputs
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val temperature: Float = 0.15f,   // lower = more structured JSON output
    val randomSeed: Int = 7,
    val promptTokenLimit: Int = 4096,  // Gemma 3n supports 32K, use 4K for agent
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

    private val json = Json { ignoreUnknownKeys = true }
    private var inference: LlmInference? = null
    private var loaded = false

    val isLoaded: Boolean
        get() = loaded

    fun updateRuntimeConfig(config: LLMRuntimeConfig) {
        runtimeConfig = config
    }

    suspend fun load() = withContext(Dispatchers.IO) {
        loadInternal()
    }

    private fun loadInternal() {
        if (loaded && inference != null) return

        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(runtimeConfig.maxOutputTokens.coerceIn(64, 2048))
            .setMaxTopK(runtimeConfig.topK.coerceIn(1, 256))
            .build()
        inference = LlmInference.createFromOptions(context, options)
        loaded = true
    }

    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        val llm = requireInference()
        runInference(llm, prompt).first
    }

    suspend fun generateDetailed(prompt: String): LLMDecisionResult = withContext(Dispatchers.IO) {
        val llm = requireInference()
        val promptTokens = tokenCountWithFallback(llm, prompt)
        val (text, latencyMs) = runInference(llm, prompt)

        val entropy = tryExtractEntropyFromInference(llm, prompt)
        if (entropy != null) {
            LLMDecisionResult(
                rawText = text,
                entropy = entropy,
                entropySource = "logprobs",
                latencyMs = latencyMs,
                promptTokens = promptTokens,
            )
        } else {
            LLMDecisionResult(
                rawText = text,
                entropy = null,
                entropySource = "not_available",
                latencyMs = latencyMs,
                promptTokens = promptTokens,
            )
        }
    }

    suspend fun decideAction(
        systemPrompt: String,
        screenContext: String,
        task: String,
        history: List<PromptTemplates.HistoryItem> = emptyList(),
        promptVersion: String = "reflect_fewshot_v1",
    ): String {
        return decideActionDetailed(
            systemPrompt = systemPrompt,
            screenContext = screenContext,
            task = task,
            history = history,
            promptVersion = promptVersion,
        ).rawText
    }

    suspend fun decideActionDetailed(
        systemPrompt: String,
        screenContext: String,
        task: String,
        history: List<PromptTemplates.HistoryItem> = emptyList(),
        promptVersion: String = "reflect_fewshot_v1",
    ): LLMDecisionResult = withContext(Dispatchers.IO) {
        val llm = requireInference()
        val resolvedSystemPrompt = if (systemPrompt.isBlank()) {
            PromptTemplates.resolveSystemPrompt(promptVersion)
        } else {
            systemPrompt
        }

        val boundedPrompt = buildPromptWithinBudget(
            task = task,
            screenContext = screenContext,
            history = history,
            promptVersion = promptVersion,
            systemPrompt = resolvedSystemPrompt,
            llm = llm,
        )

        val detail = generateDetailed(boundedPrompt.prompt)
        val normalizedJson = normalizeDecisionJson(detail.rawText)

        detail.copy(
            rawText = normalizedJson,
            promptTokens = boundedPrompt.promptTokens,
            compressionStats = boundedPrompt.compressionStats,
        )
    }

    suspend fun runDecisionBenchmark(
        iterations: Int = 5,
        warmupRounds: Int = 1,
        task: String = DEFAULT_BENCHMARK_TASK,
        screenContext: String = DEFAULT_BENCHMARK_SCREEN,
    ): GemmaBenchmarkResult = withContext(Dispatchers.IO) {
        val loadStart = System.nanoTime()
        loadInternal()
        val modelLoadMs = elapsedMs(loadStart)

        var warmupMs = 0L
        repeat(warmupRounds.coerceAtLeast(0)) {
            val detail = decideActionDetailed(
                systemPrompt = PromptTemplates.resolveSystemPrompt("baseline_v1"),
                screenContext = screenContext,
                task = task,
                history = emptyList(),
                promptVersion = "baseline_v1",
            )
            warmupMs += detail.latencyMs
        }

        val latencies = mutableListOf<Long>()
        var validJsonCount = 0
        var entropyAvailable = false
        var promptTokens = 0

        repeat(iterations.coerceAtLeast(1)) {
            val detail = decideActionDetailed(
                systemPrompt = PromptTemplates.resolveSystemPrompt("baseline_v1"),
                screenContext = screenContext,
                task = task,
                history = emptyList(),
                promptVersion = "baseline_v1",
            )
            latencies += detail.latencyMs
            promptTokens = detail.promptTokens
            if (isValidDecisionJson(detail.rawText)) validJsonCount++
            if (detail.entropySource == "logprobs") entropyAvailable = true
        }

        val sorted = latencies.sorted()
        val avg = latencies.average().toFloat()
        val p95Index = ((sorted.size - 1) * 0.95f).toInt().coerceIn(0, sorted.size - 1)
        val p95 = sorted[p95Index]
        val min = sorted.first()
        val max = sorted.last()
        val jsonRate = validJsonCount.toFloat() / latencies.size.toFloat()

        GemmaBenchmarkResult(
            modelLoadMs = modelLoadMs,
            warmupMs = warmupMs,
            avgLatencyMs = avg,
            p95LatencyMs = p95,
            minLatencyMs = min,
            maxLatencyMs = max,
            validJsonRate = jsonRate,
            logprobsAvailable = entropyAvailable,
            promptTokens = promptTokens,
            meetsLatencyTarget = avg < LATENCY_TARGET_MS,
            iterations = latencies.size,
        )
    }

    private fun tryExtractEntropyFromInference(llm: LlmInference, prompt: String): Float? {
        val cls = llm.javaClass
        val candidateNames = listOf(
            "generateResponseWithMetadata",
            "generateResponseWithLogprobs",
            "generateResponseResult",
        )

        for (method in cls.methods) {
            if (method.name !in candidateNames) continue
            if (method.parameterTypes.size != 1) continue

            try {
                val result = method.invoke(llm, prompt) ?: continue
                val logprobs = collectLogprobs(result)
                val entropy = shannonEntropyFromLogprobs(logprobs)
                if (entropy != null) return entropy
            } catch (_: Exception) {
                // Ignore reflection mismatch and keep fallback behavior.
            }
        }
        return null
    }

    private fun collectLogprobs(result: Any): List<Float> {
        val out = mutableListOf<Float>()

        fun collectFromAny(value: Any?) {
            when (value) {
                null -> Unit
                is Number -> out += value.toFloat()
                is Array<*> -> value.forEach { collectFromAny(it) }
                is Iterable<*> -> value.forEach { collectFromAny(it) }
                else -> {
                    val methods = value.javaClass.methods
                    for (name in listOf("getLogprob", "getLogProbs", "getTopLogprobs", "getTopLogProbs", "getContent")) {
                        val m = methods.firstOrNull {
                            it.name == name &&
                                it.parameterCount == 0 &&
                                it.returnType != Void.TYPE
                        } ?: continue
                        try {
                            collectFromAny(m.invoke(value))
                        } catch (_: Exception) {
                            // no-op
                        }
                    }
                }
            }
        }

        collectFromAny(result)
        return out
    }

    private fun shannonEntropyFromLogprobs(logprobs: List<Float>): Float? {
        if (logprobs.size < 2) return null

        val finite = logprobs.filter { it.isFinite() }
        if (finite.size < 2) return null

        val maxLp = finite.maxOrNull() ?: return null
        val probs = finite.map { exp((it - maxLp).toDouble()).toFloat() }
        val total = probs.sum()
        if (total <= 0f) return null
        val normalized = probs.map { it / total }

        val log2 = ln(2f)
        var entropy = 0f
        for (p in normalized) {
            if (p > 0f) entropy -= p * (ln(p) / log2)
        }
        val maxEntropy = ln(normalized.size.toFloat()) / log2
        return if (maxEntropy > 0f) (entropy / maxEntropy).coerceIn(0f, 1f) else null
    }

    private fun requireInference(): LlmInference {
        return inference ?: throw IllegalStateException("Model not loaded. Call load() first.")
    }

    private fun runInference(llm: LlmInference, prompt: String): Pair<String, Long> {
        val sessionOptions = LlmInferenceSession.LlmInferenceSessionOptions.builder()
            .setTopK(runtimeConfig.topK.coerceIn(1, 256))
            .setTopP(runtimeConfig.topP.coerceIn(0.01f, 1f))
            .setTemperature(runtimeConfig.temperature.coerceIn(0f, 2f))
            .setRandomSeed(runtimeConfig.randomSeed)
            .build()

        val session = LlmInferenceSession.createFromOptions(llm, sessionOptions)
        val startNs = System.nanoTime()
        return try {
            session.addQueryChunk(prompt)
            val output = session.generateResponse()
            output to elapsedMs(startNs)
        } finally {
            session.close()
        }
    }

    private fun buildPromptWithinBudget(
        task: String,
        screenContext: String,
        history: List<PromptTemplates.HistoryItem>,
        promptVersion: String,
        systemPrompt: String,
        llm: LlmInference,
    ): PromptBuildResult {
        var workingHistory = history
        var workingScreen = screenContext
        val originalPrompt = renderPrompt(
            task = task,
            screenContext = workingScreen,
            history = workingHistory,
            promptVersion = promptVersion,
            systemPrompt = systemPrompt,
        )
        val originalTokens = tokenCountWithFallback(llm, originalPrompt)

        var droppedHistory = 0
        var screenTrimmed = false
        var prompt = originalPrompt
        var promptTokens = originalTokens

        while (promptTokens > runtimeConfig.promptTokenLimit && workingHistory.isNotEmpty()) {
            workingHistory = workingHistory.drop(1)
            droppedHistory++
            prompt = renderPrompt(
                task = task,
                screenContext = workingScreen,
                history = workingHistory,
                promptVersion = promptVersion,
                systemPrompt = systemPrompt,
            )
            promptTokens = tokenCountWithFallback(llm, prompt)
        }

        var screenCharsBudget = max(MIN_SCREEN_CONTEXT_CHARS, workingScreen.length)
        while (promptTokens > runtimeConfig.promptTokenLimit && screenCharsBudget > MIN_SCREEN_CONTEXT_CHARS) {
            val nextBudget = max(MIN_SCREEN_CONTEXT_CHARS, (screenCharsBudget * 0.8f).toInt())
            val compact = compactScreenContext(screenContext, nextBudget)
            if (compact == workingScreen) break

            workingScreen = compact
            screenCharsBudget = nextBudget
            screenTrimmed = true

            prompt = renderPrompt(
                task = task,
                screenContext = workingScreen,
                history = workingHistory,
                promptVersion = promptVersion,
                systemPrompt = systemPrompt,
            )
            promptTokens = tokenCountWithFallback(llm, prompt)
        }

        return PromptBuildResult(
            prompt = prompt,
            promptTokens = promptTokens,
            compressionStats = PromptCompressionStats(
                originalPromptTokens = originalTokens,
                finalPromptTokens = promptTokens,
                droppedHistoryItems = droppedHistory,
                screenContextTrimmed = screenTrimmed,
            ),
        )
    }

    private fun renderPrompt(
        task: String,
        screenContext: String,
        history: List<PromptTemplates.HistoryItem>,
        promptVersion: String,
        systemPrompt: String,
    ): String {
        val messages = PromptTemplates.buildDecisionMessages(
            task = task,
            screenContext = screenContext,
            history = history,
            promptVersion = promptVersion,
        ).toMutableList()

        if (messages.isNotEmpty() && messages[0].role == "system") {
            messages[0] = messages[0].copy(content = systemPrompt)
        }
        return PromptTemplates.renderMessagesAsPrompt(messages)
    }

    private fun compactScreenContext(screenContext: String, maxChars: Int): String {
        val lines = screenContext.lineSequence().map { it.trimEnd() }.filter { it.isNotBlank() }.toList()
        if (lines.isEmpty()) return screenContext

        val header = lines.take(2)
        val indexed = lines.drop(2).filter { it.startsWith("[") }
        val priority = indexed.filter {
            it.contains("EditText") || it.contains("Button") || it.contains("clickable=true") || it.contains("editable=true")
        }
        val maxLines = (maxChars / 40).coerceAtLeast(8)
        val body = (priority + indexed).distinct().take((maxLines - header.size).coerceAtLeast(0))
        val composed = (header + body).joinToString("\n")
        if (composed.length <= maxChars) return composed
        return composed.take(maxChars)
    }

    private fun tokenCountWithFallback(llm: LlmInference, prompt: String): Int {
        val exact = runCatching { llm.sizeInTokens(prompt) }.getOrNull()
        if (exact != null && exact > 0) return exact
        return approxTokenCount(prompt)
    }

    private fun approxTokenCount(text: String): Int {
        return (text.length / APPROX_CHARS_PER_TOKEN).coerceAtLeast(1)
    }

    private fun normalizeDecisionJson(raw: String): String {
        val obj = extractJsonObject(raw)
        if (obj == null) {
            return fallbackDecisionJson("Model output is not valid JSON")
        }

        return try {
            val root = json.parseToJsonElement(obj).jsonObject
            val reasoning = root["reasoning"]?.jsonPrimitive?.contentOrNull.orEmpty()
            val action = root["action"]?.jsonPrimitive?.contentOrNull.orEmpty()
            val params = root["params"]?.jsonObject ?: JsonObject(emptyMap())

            val safeAction = action.takeIf { it in SUPPORTED_ACTIONS } ?: "wait"
            val safeParams = when (safeAction) {
                "wait" -> {
                    val ms = params["ms"]?.jsonPrimitive?.longOrNull ?: 500L
                    buildJsonObject { put("ms", JsonPrimitive(ms.coerceIn(50L, 5000L))) }
                }
                else -> params
            }

            buildJsonObject {
                put("reasoning", JsonPrimitive(reasoning.ifBlank { "auto-normalized" }))
                put("action", JsonPrimitive(safeAction))
                put("params", safeParams)
            }.toString()
        } catch (_: Exception) {
            fallbackDecisionJson("Model JSON parse failed")
        }
    }

    private fun fallbackDecisionJson(reason: String): String {
        return buildJsonObject {
            put("reasoning", JsonPrimitive(reason))
            put("action", JsonPrimitive("wait"))
            put(
                "params",
                buildJsonObject {
                    put("ms", JsonPrimitive(500))
                },
            )
        }.toString()
    }

    private fun extractJsonObject(raw: String): String? {
        val text = raw.trim()
        var depth = 0
        var start = -1
        var inString = false
        var escaped = false

        for (idx in text.indices) {
            val ch = text[idx]
            if (inString) {
                if (escaped) {
                    escaped = false
                } else if (ch == '\\') {
                    escaped = true
                } else if (ch == '"') {
                    inString = false
                }
                continue
            }

            when (ch) {
                '"' -> inString = true
                '{' -> {
                    if (depth == 0) start = idx
                    depth++
                }
                '}' -> {
                    if (depth <= 0) continue
                    depth--
                    if (depth == 0 && start >= 0) {
                        return text.substring(start, idx + 1)
                    }
                }
            }
        }
        return null
    }

    private fun isValidDecisionJson(raw: String): Boolean {
        return try {
            val parsed = json.parseToJsonElement(raw).jsonObject
            val action = parsed["action"]?.jsonPrimitive?.contentOrNull
            val params = parsed["params"]?.jsonObject
            !action.isNullOrBlank() && params != null
        } catch (_: Exception) {
            false
        }
    }

    private fun elapsedMs(startNs: Long): Long {
        return (System.nanoTime() - startNs) / 1_000_000L
    }

    fun close() {
        inference?.close()
        inference = null
        loaded = false
    }

    private data class PromptBuildResult(
        val prompt: String,
        val promptTokens: Int,
        val compressionStats: PromptCompressionStats,
    )

    companion object {
        const val MEDIAPIPE_GENAI_VERSION = "0.10.24"
        private const val APPROX_CHARS_PER_TOKEN = 4
        private const val MIN_SCREEN_CONTEXT_CHARS = 256
        private const val LATENCY_TARGET_MS = 500f
        private val SUPPORTED_ACTIONS = setOf("tap", "type_text", "scroll", "back", "home", "launch", "wait", "done")
        private const val DEFAULT_BENCHMARK_TASK = "Open Wi-Fi settings"
        private const val DEFAULT_BENCHMARK_SCREEN = """
App: com.android.settings
[0] TextView "Network & Internet"
[1] TextView "Apps"
[2] TextView "Display"
"""
    }
}
