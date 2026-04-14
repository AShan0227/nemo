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
import com.nemo.model.ModelManager
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
)

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
            "https://huggingface.co/google/gemma-1b-it/resolve/main/gemma-1b-it-q4_0.bin",
        )
    }
    var expectedSha256 by remember { mutableStateOf("") }
    var progress by remember { mutableStateOf<DownloadProgress?>(null) }
    var message by remember { mutableStateOf("") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Settings", style = MaterialTheme.typography.headlineMedium)
            TextButton(onClick = onBack) { Text("Back") }
        }

        Spacer(Modifier.height(12.dp))

        LazyColumn(modifier = Modifier.fillMaxSize()) {
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text("Model", style = MaterialTheme.typography.titleMedium)
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
                        Text("Execution", style = MaterialTheme.typography.titleMedium)
                        Text("Max steps: ${settings.maxSteps}")
                        Slider(
                            value = settings.maxSteps.toFloat(),
                            onValueChange = { onSettingsChange(settings.copy(maxSteps = it.toInt().coerceIn(5, 100))) },
                            valueRange = 5f..100f,
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
