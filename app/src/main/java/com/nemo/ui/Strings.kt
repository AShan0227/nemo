package com.nemo.ui

import java.util.Locale

/**
 * App UI strings with Chinese/English support.
 * Auto-detects system language, user can override in Settings.
 */
object Strings {
    var isChinese: Boolean = Locale.getDefault().language == "zh"

    // Main Screen
    val appTitle get() = if (isChinese) "Nemo 智能助手" else "Nemo Agent"
    val settings get() = if (isChinese) "设置" else "Settings"
    val demo get() = if (isChinese) "演示" else "Demo"
    val privacy get() = if (isChinese) "隐私" else "Privacy"
    val task get() = if (isChinese) "任务" else "Task"
    val taskPlaceholder get() = if (isChinese) "例如：打开微信给张三发消息" else "e.g. Open WeChat and send hello"
    val executeTask get() = if (isChinese) "执行任务" else "Execute Task"
    val running get() = if (isChinese) "执行中..." else "Running..."
    val ready get() = if (isChinese) "就绪" else "Ready"
    val shareResult get() = if (isChinese) "分享结果" else "Share Result"

    // Setup Wizard
    val step1Title get() = if (isChinese) "开启屏幕读取" else "Enable Screen Access"
    val step1Desc get() = if (isChinese) "Nemo 需要无障碍权限来读取屏幕和执行操作。\n所有处理在手机本地完成，数据不会离开设备。" else "Nemo needs permission to read your screen and perform actions.\nAll processing happens on-device. Your data never leaves your phone."
    val step1Button get() = if (isChinese) "打开无障碍设置" else "Open Accessibility Settings"
    val step1Waiting get() = if (isChinese) "等待开启..." else "Waiting for permission..."
    val step1Ready get() = if (isChinese) "已开启" else "Ready"
    val step1ServiceEnabled get() = if (isChinese) "服务已开启" else "Service enabled"
    val continueText get() = if (isChinese) "继续" else "Continue"

    val step2Title get() = if (isChinese) "下载 AI 模型" else "Download AI Model"
    val step2Downloading get() = if (isChinese) "正在下载 AI 模型 (~700MB)，仅需一次。" else "Downloading AI model (~700MB). This only happens once."
    val step2Done get() = if (isChinese) "模型已就绪！所有推理在手机本地运行。" else "Model ready! All inference runs locally on your device."
    val step2ModelReady get() = if (isChinese) "模型已下载" else "Model downloaded"
    val step2Failed get() = if (isChinese) "下载失败" else "Download failed"
    val retry get() = if (isChinese) "重试" else "Retry"

    val step3Title get() = if (isChinese) "准备就绪！" else "Ready to Go!"
    val step3Desc get() = if (isChinese) "Nemo 已设置完成。试试快速演示或直接开始使用。" else "Nemo is set up. Try a quick demo or start using the app."
    val tryDemo get() = if (isChinese) "试一下：\"打开手机设置\"" else "Try Demo: \"Open Settings\""
    val skipToApp get() = if (isChinese) "跳过，进入应用" else "Skip to App"

    // Sandbox
    val sandboxTitle get() = if (isChinese) "演示模式" else "Sandbox Demo"
    val sandboxDesc get() = if (isChinese) "无需任何设置即可体验 Nemo。无需无障碍权限，无需下载模型。" else "Experience Nemo without any setup. No Accessibility permission, no model download."
    val runDemo get() = if (isChinese) "▶ 开始演示" else "▶ Run Demo"
    val agentRunning get() = if (isChinese) "Agent 执行中..." else "Agent running..."
    val taskCompleted get() = if (isChinese) "任务完成！" else "Task completed!"
    val tryFullMode get() = if (isChinese) "想试试真正的效果？去设置中开启完整模式。" else "Ready to try it for real? Set up full mode in Settings."
    val back get() = if (isChinese) "返回" else "Back"
    val screen get() = if (isChinese) "屏幕" else "Screen"

    // Privacy Dashboard
    val privacyTitle get() = if (isChinese) "隐私仪表盘" else "Privacy Dashboard"
    val zeroBytesUploaded get() = if (isChinese) "零字节上传" else "Zero bytes uploaded"
    val screensProcessed get() = if (isChinese) "已处理\n屏幕" else "Screens\nProcessed"
    val elementsAnalyzed get() = if (isChinese) "已分析\n元素" else "Elements\nAnalyzed"
    val sensitiveFiltered get() = if (isChinese) "敏感字段\n已过滤" else "Sensitive Fields\nFiltered"
    val piiRedacted get() = if (isChinese) "隐私数据\n已脱敏" else "PII\nRedacted"
    val actionsBlocked get() = if (isChinese) "危险操作\n已拦截" else "Actions\nBlocked"
    val dataUploaded get() = if (isChinese) "数据\n上传" else "Data\nUploaded"

    // Settings
    val language get() = if (isChinese) "语言 / Language" else "Language / 语言"
    val switchToChinese get() = if (isChinese) "English" else "中文"
    val model get() = if (isChinese) "模型" else "Model"
    val safetyResearch get() = if (isChinese) "安全与研究" else "Safety & Research"
    val execution get() = if (isChinese) "执行参数" else "Execution"
    val maxSteps get() = if (isChinese) "最大步数" else "Max steps"
    val actionDelay get() = if (isChinese) "操作延迟" else "Action delay"

    // Accessibility
    val accessibilityRequired get() = if (isChinese) "需要开启无障碍服务" else "Accessibility Service Required"
    val accessibilityExplain get() = if (isChinese) "Nemo 需要无障碍服务来读取屏幕内容和执行操作。" else "Nemo needs Accessibility Service to read screen content and perform actions."
    val openSettings get() = if (isChinese) "打开无障碍设置" else "Open Accessibility Settings"
}
