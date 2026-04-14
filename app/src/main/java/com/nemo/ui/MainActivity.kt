package com.nemo.ui

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.nemo.agent.PhoneAgent
import com.nemo.agent.StepRecord
import com.nemo.agent.TaskStatus
import com.nemo.model.OnDeviceLLM
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                MainScreen(
                    onOpenAccessibilitySettings = {
                        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                    }
                )
            }
        }
    }
}

@Composable
fun MainScreen(onOpenAccessibilitySettings: () -> Unit) {
    var taskInput by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Ready") }
    var steps by remember { mutableStateOf(listOf<StepRecord>()) }
    val scope = rememberCoroutineScope()

    val isServiceConnected = NemoAccessibilityService.instance != null

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
    ) {
        // Header
        Text("Nemo Agent", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))

        // Service status
        if (!isServiceConnected) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text("Accessibility Service not enabled")
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = onOpenAccessibilitySettings) {
                        Text("Open Settings")
                    }
                }
            }
            Spacer(Modifier.height(16.dp))
        }

        // Task input
        OutlinedTextField(
            value = taskInput,
            onValueChange = { taskInput = it },
            label = { Text("Task") },
            placeholder = { Text("e.g. Open WeChat and send hello to John") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isRunning,
        )
        Spacer(Modifier.height(12.dp))

        // Execute button
        Button(
            onClick = {
                if (taskInput.isNotBlank() && isServiceConnected) {
                    isRunning = true
                    steps = emptyList()
                    statusText = "Running..."
                    scope.launch {
                        // TODO: initialize LLM with actual model path
                        val llm = OnDeviceLLM("/data/local/tmp/gemma-1b-q4.bin")
                        try {
                            llm.load()
                            val agent = PhoneAgent(llm)
                            val result = agent.execute(taskInput) { step ->
                                steps = steps + step
                            }
                            statusText = "${result.status.name}: ${result.summary} (${result.totalSteps} steps)"
                        } catch (e: Exception) {
                            statusText = "Error: ${e.message}"
                        } finally {
                            llm.close()
                            isRunning = false
                        }
                    }
                }
            },
            enabled = !isRunning && isServiceConnected && taskInput.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (isRunning) "Running..." else "Execute Task")
        }
        Spacer(Modifier.height(12.dp))

        // Status
        Text(statusText, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(12.dp))

        // Step log
        LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
            items(steps) { step ->
                StepCard(step)
            }
        }
    }
}

@Composable
fun StepCard(step: StepRecord) {
    val bgColor = if (step.success) Color(0xFFE8F5E9) else Color(0xFFFFEBEE)
    Card(
        colors = CardDefaults.cardColors(containerColor = bgColor),
        modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                "Step ${step.step}: ${step.action} (${step.durationMs}ms)",
                style = MaterialTheme.typography.bodyMedium,
            )
            if (step.reasoning.isNotBlank()) {
                Text(step.reasoning, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
            }
            if (step.error.isNotBlank()) {
                Text(step.error, style = MaterialTheme.typography.bodySmall, color = Color.Red)
            }
        }
    }
}
