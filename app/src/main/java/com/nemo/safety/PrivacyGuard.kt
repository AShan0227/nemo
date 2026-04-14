package com.nemo.safety

import com.nemo.screen.ScreenState
import com.nemo.screen.UIElement

/**
 * Privacy-preserving screen content filter.
 * Strips sensitive data before passing to LLM prompt.
 * All filtering happens locally — nothing leaves the device.
 */
object PrivacyGuard {

    private val SENSITIVE_CLASS_PATTERNS = setOf(
        "password", "pin", "passcode", "cvv", "cvc", "secret",
    )

    private val SENSITIVE_HINT_PATTERNS = setOf(
        "password", "密码", "pin", "passcode", "验证码",
        "cvv", "cvc", "安全码", "secret", "token",
    )

    private val PII_PATTERNS = listOf(
        Regex("\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b"),  // credit card
        Regex("\\b\\d{3}[\\s-]?\\d{2}[\\s-]?\\d{4}\\b"),               // SSN
        Regex("\\b\\d{17}[0-9Xx]\\b"),                                    // 身份证号
        Regex("\\b\\d{11}\\b"),                                           // 手机号 (11位)
    )

    /**
     * Check if a UI element is a sensitive input field.
     */
    fun isSensitiveField(element: UIElement): Boolean {
        val className = element.className.lowercase()
        val desc = element.contentDesc.lowercase()
        val resId = element.resourceId.lowercase()
        val text = element.text.lowercase()

        if (SENSITIVE_CLASS_PATTERNS.any { it in className }) return true
        if (SENSITIVE_HINT_PATTERNS.any { it in desc || it in resId || it in text }) return true
        if (element.editable && "password" in className) return true

        return false
    }

    /**
     * Redact PII patterns from text.
     */
    fun redactPII(text: String): String {
        var result = text
        for (pattern in PII_PATTERNS) {
            result = pattern.replace(result, "[REDACTED]")
        }
        return result
    }

    /**
     * Filter a ScreenState for LLM prompt — redact sensitive fields and PII.
     */
    fun filterForPrompt(screen: ScreenState): ScreenState {
        val filtered = screen.elements.map { elem ->
            if (isSensitiveField(elem)) {
                elem.copy(text = "[SENSITIVE]", contentDesc = "[SENSITIVE]")
            } else {
                elem.copy(text = redactPII(elem.text), contentDesc = redactPII(elem.contentDesc))
            }
        }
        return screen.copy(elements = filtered)
    }

    /**
     * Check if the current screen appears to be a payment/financial page.
     */
    fun isPaymentScreen(screen: ScreenState): Boolean {
        val allText = screen.elements.joinToString(" ") { "${it.text} ${it.contentDesc}" }.lowercase()
        val paymentIndicators = setOf(
            "credit card", "信用卡", "debit card", "银行卡",
            "payment method", "支付方式", "card number", "卡号",
            "expiry", "有效期", "cvv", "billing", "账单",
        )
        return paymentIndicators.count { it in allText } >= 2
    }
}
