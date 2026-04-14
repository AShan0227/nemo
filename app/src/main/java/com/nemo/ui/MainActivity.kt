package com.nemo.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.nemo.NemoApp
import com.nemo.agent.PhoneAgent
import com.nemo.agent.StepRecord
import com.nemo.service.NemoAccessibilityService
import com.nemo.service.NemoForegroundService
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestNotificationPermission()
        val app = NemoApp.instance

        setContent {
            MaterialTheme {
                val isFirstLaunch = !app.appState.setupCompleted
                if (isFirstLaunch) {
                    SetupWizard(
                        onOpenAccessibilitySettings = {
                            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                        },
                        onComplete = {
                            app.appState.setupCompleted = true
                            recreate()
                        },
                        onRunDemo = {
                            app.appState.setupCompleted = true
                            NemoForegroundService.start(this@MainActivity, "打开手机设置")
                            recreate()
                        },
                    )
                } else {
                    AppNavRoot(
                        onOpenAccessibilitySettings = {
                            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
                        },
                    )
                }
            }
        }
    }

    private fun requestNotificationPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED
            ) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1001)
            }
        }
    }
}

@Composable
private fun AppNavRoot(onOpenAccessibilitySettings: () -> Unit) {
    val app = NemoApp.instance
    val navController = rememberNavController()
    var settings by remember { mutableStateOf(app.appState.loadSettings()) }

    NavHost(navController = navController, startDestination = "main") {
        composable("main") {
            MainScreen(
                settings = settings,
                onSettingsClick = { navController.navigate("settings") },
                onSandboxClick = { navController.navigate("sandbox") },
                onPrivacyClick = { navController.navigate("privacy") },
                onOpenAccessibilitySettings = onOpenAccessibilitySettings,
            )
        }
        composable("settings") {
            SettingsScreen(
                settings = settings,
                onSettingsChange = {
                    settings = it
                    app.appState.saveSettings(it)
                },
                onBack = { navController.popBackStack() },
            )
        }
        composable("sandbox") {
            SandboxDemoScreen(onBack = { navController.popBackStack() })
        }
        composable("privacy") {
            PrivacyDashboard(
                stats = PrivacyStats(),
                onBack = { navController.popBackStack() },
            )
        }
    }
}

@Composable
fun MainScreen(
    settings: AppSettings,
    onSettingsClick: () -> Unit,
    onSandboxClick: () -> Unit = {},
    onPrivacyClick: () -> Unit = {},
    onOpenAccessibilitySettings: () -> Unit,
) {
    val app = NemoApp.instance
    var taskInput by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Ready") }
    var steps by remember { mutableStateOf(listOf<StepRecord>()) }
    val scope = rememberCoroutineScope()
    val isServiceConnected = NemoAccessibilityService.instance != null

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Nemo Agent", style = MaterialTheme.typography.headlineMedium)
            Row {
                TextButton(onClick = onSandboxClick) { Text("Demo") }
                TextButton(onClick = onPrivacyClick) { Text("Privacy") }
                TextButton(onClick = onSettingsClick) { Text("Settings") }
            }
        }
        Spacer(Modifier.height(8.dp))

        if (!isServiceConnected) {
            AccessibilityGuideCard(onOpenSettings = onOpenAccessibilitySettings)
            Spacer(Modifier.height(12.dp))
        }

        OutlinedTextField(
            value = taskInput,
            onValueChange = { taskInput = it },
            label = { Text("Task") },
            placeholder = { Text("e.g. 打开微信给张三发消息") },
            modifier = Modifier.fillMaxWidth(),
            enabled = !isRunning,
        )
        Spacer(Modifier.height(12.dp))

        Button(
            onClick = {
                if (taskInput.isBlank() || !isServiceConnected) return@Button
                isRunning = true
                steps = emptyList()
                statusText = "Running..."
                scope.launch {
                    try {
                        val llm = app.getLLM()
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
                        )
                        agent.initPersistence(app)
                        val result = agent.execute(taskInput) { step -> steps = steps + step }
                        statusText = "${result.status.name}: ${result.summary} (${result.totalSteps} steps)"
                    } catch (e: Exception) {
                        statusText = "Error: ${e.message}"
                    } finally {
                        isRunning = false
                    }
                }
            },
            enabled = !isRunning && isServiceConnected && taskInput.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (isRunning) "Running..." else "Execute Task")
        }

        Spacer(Modifier.height(8.dp))
        Text(statusText, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(8.dp))

        LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
            items(steps) { step ->
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = if (step.success) Color(0xFFE8F5E9) else Color(0xFFFFEBEE)
                    ),
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text("Step ${step.step}: ${step.action} (${step.durationMs}ms)", style = MaterialTheme.typography.bodyMedium)
                        if (step.reasoning.isNotBlank()) Text(step.reasoning, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                        if (step.error.isNotBlank()) Text(step.error, style = MaterialTheme.typography.bodySmall, color = Color.Red)
                    }
                }
            }
        }
    }
}
