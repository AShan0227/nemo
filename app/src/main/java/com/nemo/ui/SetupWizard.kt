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

@Composable
fun SetupWizard(
    onOpenAccessibilitySettings: () -> Unit,
    onComplete: () -> Unit,
    onRunDemo: () -> Unit,
) {
    val pagerState = rememberPagerState(pageCount = { 3 })
    val scope = rememberCoroutineScope()

    Column(modifier = Modifier.fillMaxSize()) {
        LinearProgressIndicator(
            progress = { (pagerState.currentPage + 1) / 3f },
            modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
        )
        HorizontalPager(state = pagerState, modifier = Modifier.weight(1f), userScrollEnabled = false) { page ->
            when (page) {
                0 -> StepAccessibility(
                    onOpenSettings = onOpenAccessibilitySettings,
                    onNext = { scope.launch { pagerState.animateScrollToPage(1) } },
                )
                1 -> StepModelDownload(onNext = { scope.launch { pagerState.animateScrollToPage(2) } })
                2 -> StepTryDemo(onRunDemo = onRunDemo, onComplete = onComplete)
            }
        }
    }
}

@Composable
private fun StepAccessibility(onOpenSettings: () -> Unit, onNext: () -> Unit) {
    var isEnabled by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        while (true) {
            isEnabled = NemoAccessibilityService.instance != null
            if (isEnabled) break
            delay(2000)
        }
    }

    WizardPage(step = 1, title = Strings.step1Title, description = Strings.step1Desc) {
        if (isEnabled) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(Strings.step1ServiceEnabled, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Text(Strings.step1Ready, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                }
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onNext, modifier = Modifier.fillMaxWidth()) { Text(Strings.continueText) }
        } else {
            val steps = if (Strings.isChinese) listOf(
                "1. 点击下面的按钮", "2. 在列表中找到 'Nemo Agent'",
                "3. 打开开关并确认", "4. 返回这里 — 自动检测",
            ) else listOf(
                "1. Tap the button below", "2. Find 'Nemo Agent' in the list",
                "3. Toggle ON and confirm", "4. Come back here — auto-detected",
            )
            steps.forEach { Text(it, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.padding(vertical = 2.dp)) }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onOpenSettings, modifier = Modifier.fillMaxWidth()) { Text(Strings.step1Button) }
            Spacer(Modifier.height(8.dp))
            Text(Strings.step1Waiting, style = MaterialTheme.typography.bodySmall, textAlign = TextAlign.Center, modifier = Modifier.fillMaxWidth())
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

    // Auto-start multi-part download from GitHub Release (no auth needed)
    LaunchedEffect(Unit) {
        if (isDone) return@LaunchedEffect
        val result = modelManager.downloadMultiPartModel(
            partUrls = ModelManager.MODEL_PART_URLS,
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
        title = Strings.step2Title,
        description = if (isDone) Strings.step2Done else Strings.step2Downloading,
    ) {
        if (isDone) {
            Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                Row(Modifier.fillMaxWidth().padding(16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(Strings.step2ModelReady, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(1f))
                    Text(Strings.step1Ready, color = MaterialTheme.colorScheme.primary, fontWeight = FontWeight.Bold)
                }
            }
            Spacer(Modifier.height(16.dp))
            Button(onClick = onNext, modifier = Modifier.fillMaxWidth()) { Text(Strings.continueText) }
        } else {
            // Progress
            progress?.let { p ->
                LinearProgressIndicator(progress = { p.percent / 100f }, modifier = Modifier.fillMaxWidth())
                Spacer(Modifier.height(8.dp))
                Text("${p.percent}% — ${p.downloadedBytes / 1_000_000}MB ${p.status}")
            }

            // Error + retry
            error?.let { e ->
                Spacer(Modifier.height(8.dp))
                Text("${Strings.step2Failed}: $e", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(8.dp))
                Button(onClick = {
                    error = null; progress = null
                    scope.launch {
                        val result = modelManager.downloadMultiPartModel(
                            partUrls = ModelManager.MODEL_PART_URLS,
                            targetPath = defaultPath,
                            onProgress = { progress = it },
                        )
                        result.fold(onSuccess = { isDone = true }, onFailure = { error = it.message })
                    }
                }) { Text(Strings.retry) }
            }

            Spacer(Modifier.height(16.dp))
            // Skip option
            OutlinedButton(onClick = onNext, modifier = Modifier.fillMaxWidth()) {
                Text(if (Strings.isChinese) "跳过（先用演示模式）" else "Skip (use Demo mode first)")
            }
        }
    }
}

@Composable
private fun StepTryDemo(onRunDemo: () -> Unit, onComplete: () -> Unit) {
    WizardPage(step = 3, title = Strings.step3Title, description = Strings.step3Desc) {
        Button(onClick = onRunDemo, modifier = Modifier.fillMaxWidth()) { Text(Strings.tryDemo) }
        Spacer(Modifier.height(12.dp))
        OutlinedButton(onClick = onComplete, modifier = Modifier.fillMaxWidth()) { Text(Strings.skipToApp) }
    }
}

@Composable
private fun WizardPage(step: Int, title: String, description: String, content: @Composable ColumnScope.() -> Unit) {
    Column(modifier = Modifier.fillMaxSize().padding(24.dp), verticalArrangement = Arrangement.Center) {
        Text("Step $step of 3", style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.primary)
        Spacer(Modifier.height(8.dp))
        Text(title, style = MaterialTheme.typography.headlineMedium)
        Spacer(Modifier.height(8.dp))
        Text(description, style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(24.dp))
        content()
    }
}
