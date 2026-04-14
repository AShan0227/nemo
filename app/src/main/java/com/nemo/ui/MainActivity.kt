package com.nemo.ui

import android.content.Intent
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.splashscreen.SplashScreen.Companion.installSplashScreen
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.nemo.agent.PhoneAgent
import com.nemo.agent.StepRecord
import com.nemo.model.ModelManager
import com.nemo.model.OnDeviceLLM
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        installSplashScreen()
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                AppNavRoot(
                    onOpenAccessibilitySettings = {
                        startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                    },
                )
            }
        }
    }
}

@Composable
private fun AppNavRoot(onOpenAccessibilitySettings: () -> Unit) {
    val navController = rememberNavController()
    val initial = defaultSettings()
    var settings by remember { mutableStateOf(initial) }

    NavHost(navController = navController, startDestination = "main") {
        composable("main") {
            MainScreen(
                settings = settings,
                onSettingsClick = { navController.navigate("settings") },
                onOpenAccessibilitySettings = onOpenAccessibilitySettings,
            )
        }
        composable("settings") {
            SettingsScreen(
                settings = settings,
                onSettingsChange = { settings = it },
                onBack = { navController.popBackStack() },
            )
        }
    }
}

@Composable
fun MainScreen(
    settings: AppSettings,
    onSettingsClick: () -> Unit,
    onOpenAccessibilitySettings: () -> Unit,
) {
    val context = LocalContext.current
    val modelManager = remember(context) { ModelManager(context) }

    var taskInput by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Ready") }
    var steps by remember { mutableStateOf(listOf<StepRecord>()) }
    val scope = rememberCoroutineScope()

    val isServiceConnected = NemoAccessibilityService.instance != null

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
    ) {
        Text("Nemo Agent", style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        TextButton(onClick = onSettingsClick) { Text("Open Settings") }
        Spacer(Modifier.height(8.dp))

        if (!isServiceConnected) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.errorContainer,
                ),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text("Accessibility Service not enabled")
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = onOpenAccessibilitySettings) {
                        Text("Open Accessibility Settings")
                    }
                }
            }
            Spacer(Modifier.height(16.dp))
        }

        OutlinedTextField(
            value = taskInput,
            onValueChange = { taskInput = it },
            label = { Text("Task") },
            placeholder = { Text("e.g. Open WeChat and send hello") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isRunning,
        )

        Spacer(Modifier.height(12.dp))

        Button(
            onClick = {
                if (taskInput.isBlank() || !isServiceConnected) return@Button
                if (!modelManager.isModelDownloaded(settings.modelPath)) {
                    statusText = "Model not found. Please download it in Settings."
                    return@Button
                }

                isRunning = true
                steps = emptyList()
                statusText = "Running..."

                scope.launch {
                    val llm = OnDeviceLLM(
                        context = context,
                        modelPath = settings.modelPath,
                        runtimeConfig = settings.toLlmRuntimeConfig(),
                    )
                    try {
                        llm.load()
                        val agent = PhoneAgent(
                            llm = llm,
                            maxSteps = settings.maxSteps,
                            actionDelayMs = settings.actionDelayMs,
                            safetyEnabled = settings.safetyEnabled,
                            homeostasisEnabled = settings.homeostasisEnabled,
                            immuneEnabled = settings.immuneEnabled,
                            inertiaEnabled = settings.inertiaEnabled,
                            explorerEnabled = settings.explorerEnabled,
                            circadianEnabled = settings.circadianEnabled,
                            contextWindowSteps = settings.contextWindowSteps,
                        )
                        val result = agent.execute(taskInput) { step ->
                            steps = steps + step
                        }
                        statusText =
                            "${result.status.name}: ${result.summary} (${result.totalSteps} steps)"
                    } catch (e: Exception) {
                        statusText = "Error: ${e.message}"
                    } finally {
                        llm.close()
                        isRunning = false
                    }
                }
            },
            enabled = !isRunning && isServiceConnected && taskInput.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (isRunning) "Running..." else "Execute Task")
        }

        Spacer(Modifier.height(12.dp))
        Text(statusText, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(12.dp))

        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f),
        ) {
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
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
    ) {
        Column(Modifier.padding(12.dp)) {
            Text(
                "Step ${step.step}: ${step.action} (${step.durationMs}ms)",
                style = MaterialTheme.typography.bodyMedium,
            )
            if (step.reasoning.isNotBlank()) {
                Text(
                    step.reasoning,
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Gray,
                )
            }
            if (step.error.isNotBlank()) {
                Text(
                    step.error,
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.Red,
                )
            }
        }
    }
}
