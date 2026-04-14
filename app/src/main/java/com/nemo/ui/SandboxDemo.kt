package com.nemo.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.nemo.agent.StepRecord
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Sandbox Demo — experience Nemo without any permissions or model.
 *
 * Uses pre-recorded screen data to simulate the full agent loop.
 * Zero setup required. Install → Tap → Watch.
 *
 * This is the "OpenClaw live demo bot" equivalent —
 * lets people experience before committing to setup.
 */

data class SandboxScenario(
    val id: String,
    val title: String,
    val description: String,
    val task: String,
    val steps: List<SandboxStep>,
)

data class SandboxStep(
    val screenDescription: String,
    val reasoning: String,
    val action: String,
    val params: Map<String, Any> = emptyMap(),
    val durationMs: Long = 150,
)

val SANDBOX_SCENARIOS = listOf(
    SandboxScenario(
        id = "settings_wifi",
        title = "Toggle WiFi",
        description = "Watch Nemo navigate to WiFi settings and toggle it",
        task = "Turn off WiFi",
        steps = listOf(
            SandboxStep("Home Screen — [0] Settings [1] Camera [2] Phone", "I need to open Settings to find WiFi", "tap", mapOf("index" to 0), 200),
            SandboxStep("Settings — [0] Network [1] WiFi [2] Bluetooth [3] Display", "I can see WiFi option, tapping it", "tap", mapOf("index" to 1), 180),
            SandboxStep("WiFi — [0] WiFi toggle (ON) [1] Network1 [2] Network2", "WiFi toggle is ON, tapping to turn OFF", "tap", mapOf("index" to 0), 150),
            SandboxStep("WiFi — [0] WiFi toggle (OFF) [1] 'WiFi is off'", "WiFi is now OFF. Task complete.", "done", mapOf("summary" to "WiFi turned off successfully"), 100),
        ),
    ),
    SandboxScenario(
        id = "wechat_message",
        title = "Send WeChat Message",
        description = "Watch Nemo open WeChat and send a message",
        task = "给张三发微信说明天开会",
        steps = listOf(
            SandboxStep("Home Screen — [0] WeChat [1] Settings [2] Camera", "I need to open WeChat to send a message", "tap", mapOf("index" to 0), 250),
            SandboxStep("WeChat — [0] Chats [1] Contacts [2] Discover [3] Me", "I need to search for 张三, going to Contacts", "tap", mapOf("index" to 1), 200),
            SandboxStep("Contacts — [0] Search [1] 张三 [2] 李四 [3] 王五", "Found 张三 in contacts list, tapping", "tap", mapOf("index" to 1), 180),
            SandboxStep("Chat: 张三 — [0] Message input [1] Send", "Now I need to type the message", "type_text", mapOf("index" to 0, "text" to "明天开会"), 300),
            SandboxStep("Chat: 张三 — [0] Message input '明天开会' [1] Send", "Message typed, tapping Send", "tap", mapOf("index" to 1), 150),
            SandboxStep("Chat: 张三 — Message sent ✓", "Message sent successfully.", "done", mapOf("summary" to "已发送微信消息给张三：明天开会"), 100),
        ),
    ),
    SandboxScenario(
        id = "taobao_search",
        title = "Search Taobao",
        description = "Watch Nemo search for products on Taobao",
        task = "在淘宝搜索 iPhone 手机壳",
        steps = listOf(
            SandboxStep("Home Screen — [0] Taobao [1] Settings [2] Camera", "Opening Taobao app", "tap", mapOf("index" to 0), 250),
            SandboxStep("Taobao Home — [0] Search bar [1] Recommendations [2] Cart", "Tapping search bar to enter query", "tap", mapOf("index" to 0), 200),
            SandboxStep("Taobao Search — [0] Search input [1] History items", "Typing search query", "type_text", mapOf("index" to 0, "text" to "iPhone 手机壳"), 300),
            SandboxStep("Taobao Search — [0] Search button [1] 'iPhone 手机壳'", "Tapping search to find products", "tap", mapOf("index" to 0), 200),
            SandboxStep("Results — [0] iPhone手机壳 ¥29 [1] 苹果手机壳 ¥19 ...", "Search results loaded. Task complete.", "done", mapOf("summary" to "淘宝搜索完成：iPhone 手机壳"), 100),
        ),
    ),
)

@Composable
fun SandboxDemoScreen(onBack: () -> Unit) {
    var selectedScenario by remember { mutableStateOf<SandboxScenario?>(null) }

    if (selectedScenario != null) {
        SandboxPlayer(
            scenario = selectedScenario!!,
            onBack = { selectedScenario = null },
        )
    } else {
        SandboxScenarioList(
            onSelectScenario = { selectedScenario = it },
            onBack = onBack,
        )
    }
}

@Composable
private fun SandboxScenarioList(
    onSelectScenario: (SandboxScenario) -> Unit,
    onBack: () -> Unit,
) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Sandbox Demo", style = MaterialTheme.typography.headlineMedium)
            TextButton(onClick = onBack) { Text("Back") }
        }
        Spacer(Modifier.height(8.dp))
        Text(
            "Experience Nemo without any setup. No Accessibility permission, no model download.",
            style = MaterialTheme.typography.bodyMedium,
            color = Color.Gray,
        )
        Spacer(Modifier.height(16.dp))

        SANDBOX_SCENARIOS.forEach { scenario ->
            Card(
                onClick = { onSelectScenario(scenario) },
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
            ) {
                Column(Modifier.padding(16.dp)) {
                    Text(scenario.title, style = MaterialTheme.typography.titleMedium)
                    Text(scenario.description, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                    Spacer(Modifier.height(4.dp))
                    Text("Task: \"${scenario.task}\"", style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                    Text("${scenario.steps.size} steps", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                }
            }
        }
    }
}

@Composable
private fun SandboxPlayer(
    scenario: SandboxScenario,
    onBack: () -> Unit,
) {
    var completedSteps by remember { mutableStateOf(listOf<StepRecord>()) }
    var currentStepIndex by remember { mutableStateOf(-1) }
    var isRunning by remember { mutableStateOf(false) }
    var currentScreen by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Column {
                Text(scenario.title, style = MaterialTheme.typography.titleLarge)
                Text("\"${scenario.task}\"", style = MaterialTheme.typography.bodyMedium, color = Color.Gray)
            }
            TextButton(onClick = onBack) { Text("Back") }
        }
        Spacer(Modifier.height(8.dp))

        // Current screen view
        if (currentScreen.isNotBlank()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = Color(0xFFF5F5F5)),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Column(Modifier.padding(12.dp)) {
                    Text("Screen", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                    Text(currentScreen, style = MaterialTheme.typography.bodyMedium)
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        // NERVE indicators
        if (isRunning) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                NerveChip("🧭 Router", "System 1.5")
                NerveChip("🛡 Safety", "ALLOW")
                NerveChip("🐜 ACO", "Learning")
            }
            Spacer(Modifier.height(8.dp))
        }

        // Run button
        Button(
            onClick = {
                if (!isRunning) {
                    isRunning = true
                    completedSteps = emptyList()
                    scope.launch {
                        for ((i, step) in scenario.steps.withIndex()) {
                            currentStepIndex = i
                            currentScreen = step.screenDescription
                            delay(step.durationMs + 500L) // simulate thinking

                            val record = StepRecord(
                                step = i + 1,
                                action = step.action,
                                params = step.params,
                                reasoning = step.reasoning,
                                success = true,
                                durationMs = step.durationMs,
                            )
                            completedSteps = completedSteps + record
                        }
                        isRunning = false
                    }
                }
            },
            enabled = !isRunning,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (isRunning) "Agent running..." else "▶ Run Demo")
        }
        Spacer(Modifier.height(8.dp))

        // Progress
        if (isRunning || completedSteps.isNotEmpty()) {
            LinearProgressIndicator(
                progress = { completedSteps.size.toFloat() / scenario.steps.size },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(4.dp))
            Text("${completedSteps.size} / ${scenario.steps.size} steps", style = MaterialTheme.typography.labelSmall)
        }
        Spacer(Modifier.height(8.dp))

        // Step log
        LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
            items(completedSteps) { step ->
                Card(
                    colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)),
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                ) {
                    Column(Modifier.padding(12.dp)) {
                        Text(
                            "Step ${step.step}: ${step.action} (${step.durationMs}ms)",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.Medium,
                        )
                        Text(step.reasoning, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                    }
                }
            }
        }

        // Completion CTA
        if (!isRunning && completedSteps.size == scenario.steps.size && completedSteps.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Column(Modifier.padding(16.dp).fillMaxWidth(), horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("Task completed!", style = MaterialTheme.typography.titleMedium)
                    Text("Ready to try it for real? Set up full mode in Settings.", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
private fun NerveChip(label: String, value: String) {
    Surface(
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.small,
    ) {
        Column(Modifier.padding(horizontal = 8.dp, vertical = 4.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(label, style = MaterialTheme.typography.labelSmall)
            Text(value, style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.Bold)
        }
    }
}
