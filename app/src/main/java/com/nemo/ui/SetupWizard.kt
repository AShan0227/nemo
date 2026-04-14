package com.nemo.ui

import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
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
    var hfToken by remember { mutableStateOf("") }
    var showTokenInput by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    // Re-check if model exists (user might have placed it manually)
    LaunchedEffect(Unit) {
        while (!isDone) {
            isDone = modelManager.isModelDownloaded(defaultPath)
            delay(3000)
        }
    }

    val modelUrl = ModelManager.DEFAULT_MODEL_URL

    WizardPage(
        step = 2,
        title = Strings.step2Title,
        description = if (isDone) Strings.step2Done else if (Strings.isChinese)
            "需要下载 Gemma AI 模型 (529MB)。\n由于 Google 许可政策，需要 HuggingFace 账号。"
        else "Need to download Gemma AI model (529MB).\nRequires HuggingFace account due to Google license.",
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
            // Option 1: Download with HuggingFace Token
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text(
                        if (Strings.isChinese) "方式一：用 HuggingFace Token 下载" else "Option 1: Download with HF Token",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Spacer(Modifier.height(8.dp))

                    if (!showTokenInput) {
                        Button(onClick = { showTokenInput = true }, modifier = Modifier.fillMaxWidth()) {
                            Text(if (Strings.isChinese) "输入 HuggingFace Token" else "Enter HuggingFace Token")
                        }
                    } else {
                        OutlinedTextField(
                            value = hfToken,
                            onValueChange = { hfToken = it },
                            label = { Text("HuggingFace Token (hf_...)") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                        )
                        Spacer(Modifier.height(8.dp))
                        Button(
                            onClick = {
                                error = null
                                scope.launch {
                                    val result = modelManager.downloadModel(
                                        url = modelUrl,
                                        targetPath = defaultPath,
                                        authToken = hfToken.trim(),
                                        onProgress = { progress = it },
                                    )
                                    result.fold(
                                        onSuccess = { isDone = true },
                                        onFailure = { error = it.message },
                                    )
                                }
                            },
                            enabled = hfToken.startsWith("hf_"),
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text(if (Strings.isChinese) "开始下载" else "Start Download")
                        }
                    }

                    progress?.let { p ->
                        Spacer(Modifier.height(8.dp))
                        LinearProgressIndicator(progress = { p.percent / 100f }, modifier = Modifier.fillMaxWidth())
                        Text("${p.percent}% — ${p.downloadedBytes / 1_000_000}MB")
                    }

                    error?.let { e ->
                        Spacer(Modifier.height(8.dp))
                        Text("${Strings.step2Failed}: $e", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            Spacer(Modifier.height(12.dp))

            // Option 2: Manual instructions
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(12.dp)) {
                    Text(
                        if (Strings.isChinese) "方式二：手动下载" else "Option 2: Manual Download",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Spacer(Modifier.height(8.dp))
                    val steps = if (Strings.isChinese) listOf(
                        "1. 在电脑上打开 huggingface.co 注册/登录",
                        "2. 搜索 \"Gemma3-1B-IT\" 并同意许可",
                        "3. 下载 gemma3-1b-it-int4.task (529MB)",
                        "4. 通过 USB 传到手机此路径：",
                    ) else listOf(
                        "1. Open huggingface.co on computer, sign up/login",
                        "2. Search \"Gemma3-1B-IT\", accept license",
                        "3. Download gemma3-1b-it-int4.task (529MB)",
                        "4. Transfer to phone at this path:",
                    )
                    steps.forEach { Text(it, style = MaterialTheme.typography.bodySmall) }
                    Spacer(Modifier.height(4.dp))
                    Text(defaultPath, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
                }
            }

            Spacer(Modifier.height(12.dp))

            // Skip option (use Sandbox demo instead)
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
