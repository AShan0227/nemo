package com.nemo.model

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.ln

/**
 * On-device LLM inference via MediaPipe + Gemma.
 */
data class LLMDecisionResult(
    val rawText: String,
    val entropy: Float? = null,
    val entropySource: String = "none",
)

class OnDeviceLLM(
    private val context: Context,
    private val modelPath: String,
) {

    private var inference: LlmInference? = null
    private var loaded = false

    val isLoaded: Boolean
        get() = loaded

    suspend fun load() = withContext(Dispatchers.IO) {
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(1024)
            .setMaxTopK(40)
            .build()
        inference = LlmInference.createFromOptions(context, options)
        loaded = true
    }

    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        val llm = inference ?: throw IllegalStateException("Model not loaded. Call load() first.")
        llm.generateResponse(prompt)
    }

    suspend fun generateDetailed(prompt: String): LLMDecisionResult = withContext(Dispatchers.IO) {
        val llm = inference ?: throw IllegalStateException("Model not loaded. Call load() first.")

        // Primary path: standard response (available on all versions).
        val text = llm.generateResponse(prompt)

        // Best-effort real logprobs extraction via reflection.
        // Some MediaPipe versions expose token metadata in advanced APIs.
        val entropy = tryExtractEntropyFromInference(llm, prompt)
        if (entropy != null) {
            LLMDecisionResult(rawText = text, entropy = entropy, entropySource = "logprobs")
        } else {
            LLMDecisionResult(rawText = text, entropy = null, entropySource = "not_available")
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
    ): LLMDecisionResult {
        val resolvedSystemPrompt = if (systemPrompt.isBlank()) {
            PromptTemplates.resolveSystemPrompt(promptVersion)
        } else {
            systemPrompt
        }

        val messages = PromptTemplates.buildDecisionMessages(
            task = task,
            screenContext = screenContext,
            history = history,
            promptVersion = promptVersion,
        ).toMutableList()

        // Ensure caller-provided system prompt wins.
        if (messages.isNotEmpty() && messages[0].role == "system") {
            messages[0] = messages[0].copy(content = resolvedSystemPrompt)
        }

        val prompt = PromptTemplates.renderMessagesAsPrompt(messages)
        return generateDetailed(prompt)
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

    fun close() {
        inference?.close()
        inference = null
        loaded = false
    }
}
