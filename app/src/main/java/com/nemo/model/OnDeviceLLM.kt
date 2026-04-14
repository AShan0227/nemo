package com.nemo.model

import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * On-device LLM inference via MediaPipe + Gemma.
 * Replaces Python's remote API client (client.py).
 *
 * Key difference: ~2500 tok/s prefill on-device vs 1-3s API latency.
 */
class OnDeviceLLM(private val modelPath: String) {

    private var inference: LlmInference? = null
    private var _loaded = false

    val isLoaded: Boolean get() = _loaded

    suspend fun load() = withContext(Dispatchers.IO) {
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelPath)
            .setMaxTokens(1024)
            .setTemperature(0.3f)
            .setTopK(40)
            .setTopP(0.95f)
            .build()
        inference = LlmInference.createFromOptions(options)
        _loaded = true
    }

    suspend fun generate(prompt: String): String = withContext(Dispatchers.IO) {
        val llm = inference ?: throw IllegalStateException("Model not loaded. Call load() first.")
        llm.generateResponse(prompt)
    }

    /**
     * Decide next action given screen context and task.
     * Returns raw LLM output (JSON string).
     */
    suspend fun decideAction(systemPrompt: String, screenContext: String, task: String): String {
        val prompt = buildString {
            appendLine(systemPrompt)
            appendLine()
            appendLine("Task: $task")
            appendLine()
            appendLine("Current Screen:")
            appendLine(screenContext)
        }
        return generate(prompt)
    }

    fun close() {
        inference?.close()
        inference = null
        _loaded = false
    }
}
