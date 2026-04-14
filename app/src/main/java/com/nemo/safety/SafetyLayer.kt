package com.nemo.safety

/**
 * Safety invariant layer — conservation-enforcing action filter.
 * Ported from Python safety/invariants.py.
 *
 * Hard constraints that can NEVER be violated by the agent.
 */

enum class Verdict { ALLOW, BLOCK, MODIFY }

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
)

typealias SafetyCheck = (SafetyContext) -> Boolean

data class SafetyRule(
    val name: String,
    val description: String,
    val check: SafetyCheck,  // returns true if safe
)

class SafetyLayer(rules: List<SafetyRule>? = null) {

    private val rules: List<SafetyRule> = rules ?: DEFAULT_RULES

    fun check(ctx: SafetyContext): SafetyResult {
        val violations = rules.filter { !it.check(ctx) }.map { it.name }
        return if (violations.isEmpty()) {
            SafetyResult(Verdict.ALLOW)
        } else {
            SafetyResult(
                verdict = Verdict.BLOCK,
                violatedRules = violations,
                message = "Blocked by: ${violations.joinToString(", ")}",
            )
        }
    }

    companion object {
        private val FINANCIAL_KW = setOf("pay", "purchase", "buy", "checkout", "transfer",
            "付款", "支付", "购买", "转账", "确认订单")
        private val PERMISSION_KW = setOf("allow", "grant", "permit", "authorize",
            "允许", "授权", "同意")
        private val DELETE_KW = setOf("delete", "remove", "clear all", "erase", "uninstall",
            "删除", "移除", "清除", "卸载")
        private val MESSAGE_KW = setOf("send", "发送", "发消息")

        val DEFAULT_RULES = listOf(
            SafetyRule("financial_guard", "No financial transactions") { ctx ->
                val text = ctx.screenText.lowercase()
                FINANCIAL_KW.none { it in text }
            },
            SafetyRule("permission_guard", "No permission grants") { ctx ->
                val text = ctx.screenText.lowercase()
                PERMISSION_KW.none { it in text }
            },
            SafetyRule("deletion_guard", "No destructive deletions") { ctx ->
                val text = ctx.screenText.lowercase()
                DELETE_KW.none { it in text }
            },
            SafetyRule("step_limit", "Must not exceed max steps") { ctx ->
                ctx.stepCount < ctx.maxSteps
            },
        )
    }
}
