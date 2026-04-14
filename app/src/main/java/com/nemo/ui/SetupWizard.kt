package com.nemo.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.nemo.model.DownloadProgress
import com.nemo.model.ModelManager
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * 3-step first-launch wizard:
 * 1. Enable Accessibility Service
 * 2. Download AI model (auto-start)
 * 3. Try a demo task
 */
@Composable
fun SetupWizard(
    onOpenAccessibilitySettings: () -> Unit,
    onComplete: () -> Unit,
    onRunDemo: () -> Unit,
) {
    val pagerState = rememberPagerState(pageCount = { 3 })
    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize()) {
        // Progress indicator
        LinearProgressIndicator(
            progress = { (pagerState.currentPage + 1) / 3f },
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
        )

        HorizontalPager(
            state = pagerState,
            modifier = Modifier.weight(1f),
            userScrollEnabled = false,
        ) { page ->
            when (page) {
                0 -> StepAccessibility(
                    onOpenSettings = onOpenAccessibilitySettings,
                    onNext = { scope.launch { pagerState.animateScrollToPage(1) } },
                )
                1 -> StepModelDownload(
                    onNext = { scope.launch { pagerState.animateScrollToPage(2) } },
                )
                2 -> StepTryDemo(
                    onRunDemo = onRunDemo,
                    onComplete = onComplete,
                )
            }
        }
    }
}

@Composable
private fun StepAccessibility(
    onOpenSettings: () -> Unit,
    onNext: () -> Unit,
) {
    var isEnabled by remember { mutableStateOf(false) }

    // Poll for accessibility service status every 2 seconds
    LaunchedEffect(Unit) {
        while (true) {
            isEnabled = NemoAccessibilityService.instance != null
            if (isEnabled) break
            delay(2000)
        }
    }

    WizardPage(
        step = 1,
        title = "Enable Screen Access",
        description = "Nemo needs permission to read your screen and perform actions.\nAll processing happens on-device. Your data never leaves your phone.",
    ) {
        if (isEnabled) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("Service enabled", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Text("Ready", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                }
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onNext, modifier = Modifier.fillMaxWidth()) {
                Text("Continue")
            }
        } else {
            val steps = listOf(
                "1. Tap the button below",
                "2. Find 'Nemo Agent' in the list",
                "3. Toggle ON and confirm",
                "4. Come back here — we'll detect it automatically",
            )
            steps.forEach {
                Text(it, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(vertical = 2.dp))
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onOpenSettings, modifier = Modifier.fillMaxWidth()) {
                Text("Open Accessibility Settings")
            }
            Spacer(Modifier.height(8.dp))
            Text(
                "Waiting for permission...",
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun StepModelDownload(onNext: () -> Unit) {
    val context = LocalContext.current
    val modelManager = remember { ModelManager(context) }
    val defaultPath = remember { modelManager.defaultModelPath() }

    var progress by remember { mutableStateOf<DownloadProgress?>(null) }
    var isDone by remember { mutableStateOf(modelManager.isModelDownloaded(defaultPath)) }
    var error by remember { mutableStateOf<String?>(null) }
    val scope = rememberCoroutineScope()

    // Auto-start download if model not present
    LaunchedEffect(Unit) {
        if (isDone) return@LaunchedEffect
        val result = modelManager.downloadModel(
            url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-q4.bin",
            targetPath = defaultPath,
            onProgress = { progress = it },
        )
        result.fold(
            onSuccess = { isDone = true },
            onFailure = { error = it.message },
        )
    }

    WizardPage(
        step = 2,
        title = "Download AI Model",
        description = if (isDone) "Model ready! All inference runs locally on your device." else "Downloading Gemma 1B (529MB). This only happens once.",
    ) {
        if (isDone) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("Model downloaded", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Text("Ready", color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                }
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onNext, modifier = Modifier.fillMaxWidth()) {
                Text("Continue")
            }
        } else {
            progress?.let { p ->
                LinearProgressIndicator(
                    progress = { p.percent / 100f },
                    modifier = Modifier.fillMaxWidth(),
                )
                Spacer(Modifier.height(8.dp))
                Text("${p.percent}% — ${p.downloadedBytes / 1_000_000}MB downloaded")
            }
            error?.let { e ->
                Spacer(Modifier.height(8.dp))
                Text("Download failed: $e", color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(8.dp))
                Button(onClick = {
                    error = null
                    scope.launch {
                        val result = modelManager.downloadModel(
                            url = "https://huggingface.co/google/gemma-3n-E2B-it-litert-lm/resolve/main/gemma-3n-E2B-it-q4.bin",
                            targetPath = defaultPath,
                            onProgress = { progress = it },
                        )
                        result.fold(
                            onSuccess = { isDone = true },
                            onFailure = { error = it.message },
                        )
                    }
                }) {
                    Text("Retry")
                }
            }
        }
    }
}

@Composable
private fun StepTryDemo(
    onRunDemo: () -> Unit,
    onComplete: () -> Unit,
) {
    WizardPage(
        step = 3,
        title = "Ready to Go!",
        description = "Nemo is set up. Try a quick demo or start using the app.",
    ) {
        Button(onClick = onRunDemo, modifier = Modifier.fillMaxWidth()) {
            Text("Try Demo: \"Open Settings\"")
        }
        Spacer(Modifier.height(12.dp))
        OutlinedButton(onClick = onComplete, modifier = Modifier.fillMaxWidth()) {
            Text("Skip to App")
        }
    }
}

@Composable
private fun WizardPage(
    step: Int,
    title: String,
    description: String,
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Text("Step $step of 3", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(8.dp))
        Text(title, style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text(description, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(24.dp))
        content()
    }
}
