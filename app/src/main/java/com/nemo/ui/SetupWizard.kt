package com.nemo.ui

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
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
    var mode by remember { mutableStateOf("choose") } // "choose", "auto", "manual"
    var copied by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    // Continuously check if model file appeared (for manual transfer)
    LaunchedEffect(Unit) {
        while (true) {
            if (modelManager.isModelDownloaded(defaultPath)) {
                isDone = true
                break
            }
            delay(3000)
        }
    }

    WizardPage(
        step = 2,
        title = Strings.step2Title,
        description = if (isDone) Strings.step2Done
            else if (Strings.isChinese) "需要下载 AI 模型才能运行。请选择下载方式。"
            else "AI model needed. Choose how to download.",
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
        } else when (mode) {

            // === Choose mode ===
            "choose" -> {
                // Option 1: Auto download
                Card(onClick = { mode = "auto" }, modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            if (Strings.isChinese) "📱 手机直接下载" else "📱 Download on phone",
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Text(
                            if (Strings.isChinese) "约 2.5GB，WiFi 下 10-20 分钟" else "~2.5GB, 10-20 min on WiFi",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Spacer(Modifier.height(12.dp))

                // Option 2: PC download + transfer
                Card(onClick = { mode = "manual" }, modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            if (Strings.isChinese) "💻 电脑下载后传到手机（推荐）" else "💻 Download on PC, transfer (recommended)",
                            style = MaterialTheme.typography.titleMedium,
                        )
                        Text(
                            if (Strings.isChinese) "速度更快，电脑下载后通过 USB 或微信传输" else "Faster — download on PC, transfer via USB or chat app",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                Spacer(Modifier.height(16.dp))
                OutlinedButton(onClick = onNext, modifier = Modifier.fillMaxWidth()) {
                    Text(if (Strings.isChinese) "跳过（先用演示模式）" else "Skip (use Demo mode first)")
                }
            }

            // === Auto download mode ===
            "auto" -> {
                // Start download
                LaunchedEffect(Unit) {
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

                progress?.let { p ->
                    LinearProgressIndicator(progress = { p.percent / 100f }, modifier = Modifier.fillMaxWidth())
                    Spacer(Modifier.height(8.dp))
                    Text("${p.percent}% — ${p.downloadedBytes / 1_000_000}MB ${p.status}")
                    Spacer(Modifier.height(4.dp))
                    Text(
                        if (Strings.isChinese) "请保持 App 在前台，不要切换网络" else "Keep app in foreground, don't switch networks",
                        style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error,
                    )
                }

                error?.let { e ->
                    Spacer(Modifier.height(8.dp))
                    Text("${Strings.step2Failed}: $e", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(8.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
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
                        OutlinedButton(onClick = { mode = "manual" }) {
                            Text(if (Strings.isChinese) "换电脑下载" else "Try PC instead")
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))
                TextButton(onClick = { mode = "choose" }) {
                    Text(if (Strings.isChinese) "← 返回选择" else "← Back to options")
                }
            }

            // === Manual PC download mode ===
            "manual" -> {
                val downloadUrl = "https://huggingface.co/litert-community/Gemma3-1B-IT"

                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            if (Strings.isChinese) "步骤：" else "Steps:",
                            style = MaterialTheme.typography.titleSmall,
                        )
                        Spacer(Modifier.height(8.dp))

                        val steps = if (Strings.isChinese) listOf(
                            "1. 在电脑浏览器打开下面的链接",
                            "2. 登录 HuggingFace（免费注册）",
                            "3. 同意 Gemma 许可协议",
                            "4. 点 Files → 下载这个文件：",
                            "   Gemma3-1B-IT_seq128_q4_block128_ekv1280.task",
                            "   （676MB）",
                            "5. 下载完后用 USB 或微信传到手机",
                            "6. 传完后这里会自动检测到 ✓",
                        ) else listOf(
                            "1. Open the link below on your PC browser",
                            "2. Log in to HuggingFace (free signup)",
                            "3. Accept the Gemma license",
                            "4. Click Files → download this file:",
                            "   Gemma3-1B-IT_seq128_q4_block128_ekv1280.task",
                            "   (676MB)",
                            "5. Transfer to phone via USB or chat app",
                            "6. This page auto-detects when ready ✓",
                        )
                        steps.forEach {
                            Text(it, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(vertical = 1.dp))
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))

                // Copy link button
                Button(onClick = {
                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    clipboard.setPrimaryClip(ClipData.newPlainText("url", downloadUrl))
                    copied = true
                }, modifier = Modifier.fillMaxWidth()) {
                    Text(if (copied) {
                        if (Strings.isChinese) "✓ 链接已复制" else "✓ Link copied"
                    } else {
                        if (Strings.isChinese) "复制下载链接" else "Copy download link"
                    })
                }

                Spacer(Modifier.height(8.dp))

                // Model file path info
                Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                    Column(Modifier.padding(12.dp)) {
                        Text(
                            if (Strings.isChinese) "传到手机后，放到以下路径：" else "After transfer, place file at:",
                            style = MaterialTheme.typography.labelSmall,
                        )
                        Text(defaultPath, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.height(4.dp))
                        Text(
                            if (Strings.isChinese) "或者直接通过微信传文件，App 会自动扫描常见目录" else "Or just send via chat app — we scan common folders",
                            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }

                Spacer(Modifier.height(8.dp))

                Text(
                    if (Strings.isChinese) "⏳ 正在等待模型文件..." else "⏳ Waiting for model file...",
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    modifier = Modifier.fillMaxWidth(),
                )

                Spacer(Modifier.height(12.dp))
                TextButton(onClick = { mode = "choose" }) {
                    Text(if (Strings.isChinese) "← 返回选择" else "← Back to options")
                }
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
