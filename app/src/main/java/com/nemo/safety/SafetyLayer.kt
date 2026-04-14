package com.nemo.safety

import com.nemo.screen.ScreenState

/**
 * Safety invariant layer — multi-level action filter.
 *
 * Level 1: Keyword-based blocking (expanded Chinese + English)
 * Level 2: Sensitive screen detection (payment, password)
 * Level 3: Rate limiting (max actions per minute)
 * Level 4: User confirmation for high-risk actions
 */

enum class Verdict { ALLOW, BLOCK, CONFIRM }

data class SafetyResult(
    val verdict: Verdict,
    val violatedRules: List<String> = emptyList(),
    val message: String = "",
)

data class SafetyContext(
    val screenText: String,
    val stepCount: Int,
    val maxSteps: Int,
    val actionName: String = "",
    val screen: ScreenState? = null,
)

typealias SafetyCheck = (SafetyContext) -> Boolean

data class SafetyRule(
    val name: String,
    val description: String,
    val check: SafetyCheck,
    val level: Verdict = Verdict.BLOCK,
)

class SafetyLayer(rules: List<SafetyRule>? = null) {

    private val rules: List<SafetyRule> = rules ?: DEFAULT_RULES
    private var lastActionTime = 0L
    private val MIN_ACTION_INTERVAL_MS = 200L

    fun check(ctx: SafetyContext): SafetyResult {
        // Rate limit check
        val now = System.currentTimeMillis()
        if (now - lastActionTime < MIN_ACTION_INTERVAL_MS) {
            return SafetyResult(Verdict.BLOCK, listOf("rate_limit"), "Too fast, waiting...")
        }
        lastActionTime = now

        val violations = mutableListOf<String>()
        var highestLevel = Verdict.ALLOW

        for (rule in rules) {
            if (!rule.check(ctx)) {
                violations.add(rule.name)
                if (rule.level == Verdict.BLOCK) highestLevel = Verdict.BLOCK
                else if (rule.level == Verdict.CONFIRM && highestLevel != Verdict.BLOCK) highestLevel = Verdict.CONFIRM
            }
        }

        // Sensitive screen detection via PrivacyGuard
        if (ctx.screen != null && PrivacyGuard.isPaymentScreen(ctx.screen)) {
            violations.add("payment_screen_guard")
            highestLevel = Verdict.BLOCK
        }

        return if (violations.isEmpty()) {
            SafetyResult(Verdict.ALLOW)
        } else {
            SafetyResult(
                verdict = highestLevel,
                violatedRules = violations,
                message = "Blocked by: ${violations.joinToString(", ")}",
            )
        }
    }

    companion object {
        // Expanded keyword sets with Chinese + English + common variants
        private val FINANCIAL_KW = setOf(
            "pay", "purchase", "buy", "checkout", "transfer", "charge", "debit",
            "payment", "billing", "order", "subscribe", "confirm order", "process payment",
            "付款", "支付", "购买", "转账", "确认订单", "结账", "扣款", "充值", "下单",
        )
        private val PERMISSION_KW = setOf(
            "allow", "grant", "permit", "authorize", "accept", "agree", "confirm access",
            "允许", "授权", "同意", "确认", "开启权限",
        )
        private val DELETE_KW = setOf(
            "delete", "remove", "clear all", "erase", "uninstall", "discard",
            "destroy", "wipe", "empty", "format", "reset",
            "删除", "移除", "清除", "卸载", "清空", "擦除", "格式化", "恢复出厂",
        )
        private val MESSAGE_KW = setOf(
            "send", "post", "publish", "submit", "broadcast", "forward",
            "发送", "发布", "提交", "转发", "群发",
        )
        private val INSTALL_KW = setOf(
            "install", "download app", "sideload",
            "安装", "下载应用",
        )

        val DEFAULT_RULES = listOf(
            SafetyRule("financial_guard", "Block financial transactions", { ctx ->
                val text = ctx.screenText.lowercase()
                FINANCIAL_KW.none { it in text }
            }, Verdict.BLOCK),

            SafetyRule("permission_guard", "Block permission grants", { ctx ->
                val text = ctx.screenText.lowercase()
                PERMISSION_KW.none { it in text }
            }, Verdict.BLOCK),

            SafetyRule("deletion_guard", "Block destructive deletions", { ctx ->
                val text = ctx.screenText.lowercase()
                DELETE_KW.none { it in text }
            }, Verdict.BLOCK),

            SafetyRule("message_guard", "Confirm before sending messages", { ctx ->
                val text = ctx.screenText.lowercase()
                MESSAGE_KW.none { it in text }
            }, Verdict.CONFIRM),

            SafetyRule("install_guard", "Block app installations", { ctx ->
                val text = ctx.screenText.lowercase()
                INSTALL_KW.none { it in text }
            }, Verdict.BLOCK),

            SafetyRule("step_limit", "Must not exceed max steps", { ctx ->
                ctx.stepCount < ctx.maxSteps
            }, Verdict.BLOCK),

            SafetyRule("password_guard", "Block actions on password fields", { ctx ->
                val text = ctx.screenText.lowercase()
                !("password" in text || "密码" in text || "pin" in text || "验证码" in text)
                    || ctx.actionName != "type_text"
            }, Verdict.BLOCK),
        )
    }
}
