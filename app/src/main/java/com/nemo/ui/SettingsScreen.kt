package com.nemo.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.nemo.model.DownloadProgress
import com.nemo.model.GemmaBenchmarkResult
import com.nemo.model.LLMRuntimeConfig
import com.nemo.model.ModelManager
import com.nemo.model.OnDeviceLLM
import kotlinx.coroutines.launch

/**
 * Runtime app settings for agent execution.
 */
data class AppSettings(
    val modelPath: String,
    val safetyEnabled: Boolean = true,
    val homeostasisEnabled: Boolean = true,
    val immuneEnabled: Boolean = false,
    val inertiaEnabled: Boolean = true,
    val explorerEnabled: Boolean = false,
    val circadianEnabled: Boolean = true,
    val maxSteps: Int = 30,
    val actionDelayMs: Long = 300,
    val contextWindowSteps: Int = 4,
    val llmMaxOutputTokens: Int = 192,
    val llmTopK: Int = 40,
    val llmTopP: Float = 0.9f,
    val llmTemperature: Float = 0.2f,
    val promptTokenLimit: Int = 1024,
)

fun AppSettings.toLlmRuntimeConfig(): LLMRuntimeConfig {
    return LLMRuntimeConfig(
        maxOutputTokens = llmMaxOutputTokens,
        topK = llmTopK,
        topP = llmTopP,
        temperature = llmTemperature,
        promptTokenLimit = promptTokenLimit,
    )
}

@Composable
fun defaultSettings(): AppSettings {
    val context = LocalContext.current
    val manager = remember(context) { ModelManager(context) }
    return remember {
        AppSettings(modelPath = manager.defaultModelPath())
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settings: AppSettings,
    onSettingsChange: (AppSettings) -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val manager = remember(context) { ModelManager(context) }
    val scope = rememberCoroutineScope()

    var downloadUrl by remember {
        mutableStateOf(
            "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task",
        )
    }
    var expectedSha256 by remember { mutableStateOf("") }
    var progress by remember { mutableStateOf<DownloadProgress?>(null) }
    var message by remember { mutableStateOf("") }
    var benchmarkRunning by remember { mutableStateOf(false) }
    var benchmarkResult by remember { mutableStateOf<GemmaBenchmarkResult?>(null) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(Strings.settings, style = MaterialTheme.typography.headlineMedium)
            TextButton(onClick = onBack) { Text(Strings.back) }
        }

        Spacer(Modifier.height(12.dp))

        LazyColumn(modifier = Modifier.fillMaxSize()) {
            // Language toggle
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(Strings.language, style = MaterialTheme.typography.titleMedium)
                        Button(onClick = {
                            Strings.isChinese = !Strings.isChinese
                        }) {
                            Text(Strings.switchToChinese)
                        }
                    }
                }
                Spacer(Modifier.height(12.dp))
            }
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text(Strings.model, style = MaterialTheme.typography.titleMedium)
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = settings.modelPath,
                            onValueChange = { onSettingsChange(settings.copy(modelPath = it)) },
                            label = { Text("Model path") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = downloadUrl,
                            onValueChange = { downloadUrl = it },
                            label = { Text("Model URL") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = expectedSha256,
                            onValueChange = { expectedSha256 = it },
                            label = { Text("SHA256 (optional)") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )

                        Spacer(Modifier.height(8.dp))
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = {
                                onSettingsChange(settings.copy(modelPath = manager.defaultModelPath()))
                                message = "Reset to default private model path"
                            }) {
                                Text("Use Default Path")
                            }
                            Button(onClick = {
                                scope.launch {
                                    message = ""
                                    val result = manager.downloadModel(
                                        url = downloadUrl,
                                        targetPath = settings.modelPath,
                                        expectedSha256 = expectedSha256.ifBlank { null },
                                        onProgress = { progress = it },
                                    )
                                    message = result.fold(
                                        onSuccess = { file -> "Model ready: ${file.absolutePath}" },
                                        onFailure = { err -> "Download failed: ${err.message}" },
                                    )
                                }
                            }) {
                                Text("Download")
                            }
                        }

                        progress?.let { p ->
                            Spacer(Modifier.height(8.dp))
                            LinearProgressIndicator(
                                progress = { p.percent / 100f },
                                modifier = Modifier.fillMaxWidth(),
                            )
                            Text("${p.status}: ${p.percent}% (${p.downloadedBytes}/${p.totalBytes})")
                        }

                        if (message.isNotBlank()) {
                            Spacer(Modifier.height(8.dp))
                            Text(message)
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Safety & Research", style = MaterialTheme.typography.titleMedium)
                        ToggleRow(
                            label = "Safety",
                            checked = settings.safetyEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(safetyEnabled = it)) },
                        )
                        ToggleRow(
                            label = "Homeostasis",
                            checked = settings.homeostasisEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(homeostasisEnabled = it)) },
                        )
                        ToggleRow(
                            label = "Immune",
                            checked = settings.immuneEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(immuneEnabled = it)) },
                        )
                        ToggleRow(
                            label = "Inertia",
                            checked = settings.inertiaEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(inertiaEnabled = it)) },
                        )
                        ToggleRow(
                            label = "Explorer",
                            checked = settings.explorerEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(explorerEnabled = it)) },
                        )
                        ToggleRow(
                            label = "Circadian",
                            checked = settings.circadianEnabled,
                            onCheckedChange = { onSettingsChange(settings.copy(circadianEnabled = it)) },
                        )
                    }
                }

                Spacer(Modifier.height(12.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("LLM Tuning", style = MaterialTheme.typography.titleMedium)
                        Text("MediaPipe tasks-genai ${OnDeviceLLM.MEDIAPIPE_GENAI_VERSION}")

                        Text("Max output tokens: ${settings.llmMaxOutputTokens}")
                        Slider(
                            value = settings.llmMaxOutputTokens.toFloat(),
                            onValueChange = {
                                onSettingsChange(
                                    settings.copy(llmMaxOutputTokens = it.toInt().coerceIn(64, 512)),
                                )
                            },
                            valueRange = 64f..512f,
                        )

                        Text("Top-K: ${settings.llmTopK}")
                        Slider(
                            value = settings.llmTopK.toFloat(),
                            onValueChange = {
                                onSettingsChange(settings.copy(llmTopK = it.toInt().coerceIn(1, 128)))
                            },
                            valueRange = 1f..128f,
                        )

                        Text("Top-P: ${"%.2f".format(settings.llmTopP)}")
                        Slider(
                            value = settings.llmTopP,
                            onValueChange = {
                                onSettingsChange(settings.copy(llmTopP = it.coerceIn(0.5f, 1f)))
                            },
                            valueRange = 0.5f..1f,
                        )

                        Text("Temperature: ${"%.2f".format(settings.llmTemperature)}")
                        Slider(
                            value = settings.llmTemperature,
                            onValueChange = {
                                onSettingsChange(settings.copy(llmTemperature = it.coerceIn(0f, 1.5f)))
                            },
                            valueRange = 0f..1.5f,
                        )

                        Text("Prompt token limit: ${settings.promptTokenLimit}")
                        Slider(
                            value = settings.promptTokenLimit.toFloat(),
                            onValueChange = {
                                onSettingsChange(
                                    settings.copy(promptTokenLimit = it.toInt().coerceIn(512, 2048)),
                                )
                            },
                            valueRange = 512f..2048f,
                        )

                        Button(
                            onClick = {
                                if (!manager.isModelDownloaded(settings.modelPath)) {
                                    message = "Model not found. Download Gemma first."
                                    return@Button
                                }
                                scope.launch {
                                    benchmarkRunning = true
                                    benchmarkResult = null
                                    val llm = OnDeviceLLM(
                                        context = context,
                                        modelPath = settings.modelPath,
                                        runtimeConfig = settings.toLlmRuntimeConfig(),
                                    )
                                    try {
                                        llm.load()
                                        benchmarkResult = llm.runDecisionBenchmark(iterations = 5, warmupRounds = 1)
                                        message = if (benchmarkResult?.meetsLatencyTarget == true) {
                                            "Benchmark passed (<500ms avg)."
                                        } else {
                                            "Benchmark finished (>=500ms avg). Tune tokens/top-k/top-p."
                                        }
                                    } catch (e: Exception) {
                                        message = "Benchmark failed: ${e.message}"
                                    } finally {
                                        llm.close()
                                        benchmarkRunning = false
                                    }
                                }
                            },
                            enabled = !benchmarkRunning,
                        ) {
                            Text(if (benchmarkRunning) "Benchmarking..." else "Run Gemma Benchmark")
                        }

                        benchmarkResult?.let { r ->
                            Text("Load: ${r.modelLoadMs}ms, Warmup: ${r.warmupMs}ms")
                            Text(
                                "Latency avg/p95: ${"%.1f".format(r.avgLatencyMs)}ms / ${r.p95LatencyMs}ms",
                            )
                            Text("Min/Max: ${r.minLatencyMs}ms / ${r.maxLatencyMs}ms")
                            Text(
                                "JSON valid rate: ${(r.validJsonRate * 100f).toInt()}%, prompt tokens: ${r.promptTokens}",
                            )
                            Text(
                                if (r.logprobsAvailable) {
                                    "Logprobs: available"
                                } else {
                                    "Logprobs: not available in current MediaPipe API"
                                },
                            )
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Execution", style = MaterialTheme.typography.titleMedium)
                        Text("Max steps: ${settings.maxSteps}")
                        Slider(
                            value = settings.maxSteps.toFloat(),
                            onValueChange = { onSettingsChange(settings.copy(maxSteps = it.toInt().coerceIn(5, 100))) },
                            valueRange = 5f..100f,
                        )
                        Text("Context window steps: ${settings.contextWindowSteps}")
                        Slider(
                            value = settings.contextWindowSteps.toFloat(),
                            onValueChange = {
                                onSettingsChange(
                                    settings.copy(contextWindowSteps = it.toInt().coerceIn(1, 8)),
                                )
                            },
                            valueRange = 1f..8f,
                        )
                        Text("Action delay: ${settings.actionDelayMs} ms")
                        Slider(
                            value = settings.actionDelayMs.toFloat(),
                            onValueChange = {
                                onSettingsChange(
                                    settings.copy(actionDelayMs = it.toLong().coerceIn(100L, 3000L)),
                                )
                            },
                            valueRange = 100f..3000f,
                        )
                        Text(
                            "Estimated inference latency: ${settings.actionDelayMs} ms + model time",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ToggleRow(label: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label)
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
