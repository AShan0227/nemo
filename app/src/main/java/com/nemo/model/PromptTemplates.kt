package com.nemo.model

/**
 * Prompt templates and prompt-building helpers for on-device inference.
 * Mirrors Python src/llm/prompts.py behavior.
 */
object PromptTemplates {

    data class Message(val role: String, val content: String)

    data class HistoryItem(
        val step: Int,
        val action: String,
        val success: Boolean,
        val error: String = "",
    )

    const val BASELINE_SYSTEM_PROMPT: String = """
You are an AI phone assistant that controls an Android device.
Pick EXACTLY ONE next action.

Rules:
1. Only use element indices visible on the current screen.
2. Prefer reversible low-risk actions when uncertain.
3. Use type_text with both index and text.
4. Use done only when the task is truly complete.
"""

    const val REFLECT_SYSTEM_PROMPT: String = """
You are an AI phone assistant that controls an Android device to complete user tasks.

Decision protocol (reflection-before-action):
1. Reflect briefly on user intent, current progress, and risk.
2. Pick EXACTLY ONE next action that is safest and most useful now.
3. If uncertain, prefer evidence-gathering actions (scroll, wait, back).

Hard constraints:
1. ONLY reference element indices shown in the current screen.
2. NEVER perform payments, permission grants, destructive deletes,
   or irreversible operations without explicit confirmation.
3. For text input, use type_text with both index and text.
4. For completion, use done with a concise factual summary.

Fallback output format (strict JSON):
{"reasoning":"...","action":"...","params":{...}}
"""

    private val FEW_SHOT_EXAMPLES: List<Message> = listOf(
        Message(
            role = "user",
            content = """
Task: Send "到了" to Xiao Wang in WeChat.

Current Screen:
App: com.tencent.mm
[0] EditText "Message"
[1] Button "Send"
""".trimIndent(),
        ),
        Message(
            role = "assistant",
            content = """{"reasoning":"Need to type the requested message first.","action":"type_text","params":{"index":0,"text":"到了"}}""",
        ),
        Message(
            role = "user",
            content = """
Task: Open Wi-Fi settings.

Current Screen:
App: com.android.settings
[0] TextView "Network & Internet"
[1] TextView "Apps"
""".trimIndent(),
        ),
        Message(
            role = "assistant",
            content = """{"reasoning":"Network settings are the direct path to Wi-Fi.","action":"tap","params":{"index":0}}""",
        ),
        Message(
            role = "user",
            content = """
Task: Find the refund option.

Current Screen:
App: com.shopping.app
[0] TextView "Order details"
[1] TextView "Shipping"
""".trimIndent(),
        ),
        Message(
            role = "assistant",
            content = """{"reasoning":"Refund option is not visible yet, gather more evidence.","action":"scroll","params":{"direction":"down","amount":1}}""",
        ),
    )

    fun resolveSystemPrompt(promptVersion: String?): String {
        return if (promptVersion == "baseline_v1") BASELINE_SYSTEM_PROMPT else REFLECT_SYSTEM_PROMPT
    }

    fun buildDecisionMessages(
        task: String,
        screenContext: String,
        history: List<HistoryItem> = emptyList(),
        promptVersion: String = "reflect_fewshot_v1",
    ): List<Message> {
        val messages = mutableListOf<Message>()
        messages += Message("system", resolveSystemPrompt(promptVersion))

        if (promptVersion != "baseline_v1") {
            messages += FEW_SHOT_EXAMPLES
        }

        val user = buildString {
            appendLine("Task: $task")
            appendLine()
            appendLine("Current Screen:")
            appendLine(screenContext)
            val historyBlock = formatHistoryBlock(history)
            if (historyBlock.isNotBlank()) {
                appendLine()
                appendLine(historyBlock)
            }
            appendLine()
            appendLine("Return exactly one next action.")
        }
        messages += Message("user", user.trim())
        return messages
    }

    /**
     * MediaPipe LLM inference is prompt-string based; flatten chat messages into one prompt.
     */
    fun renderMessagesAsPrompt(messages: List<Message>): String {
        return buildString {
            for (m in messages) {
                appendLine("<${m.role}>")
                appendLine(m.content.trim())
                appendLine("</${m.role}>")
                appendLine()
            }
        }.trim()
    }

    fun buildPlanningPrompt(task: String, currentApp: String, screenSummary: String): String {
        return """
Given the user's task and current screen, create a step-by-step plan.

Task: $task
Current App: $currentApp
Current Screen: $screenSummary

Provide a plan as a JSON array of steps:
[
  {
    "step": 1,
    "action_name": "tap|type_text|scroll|back|home|launch|wait|done",
    "params": {"index": 0},
    "action": "short description",
    "expected_result": "what should happen"
  }
]
""".trimIndent()
    }

    private fun formatHistoryBlock(history: List<HistoryItem>): String {
        if (history.isEmpty()) return ""
        val lines = mutableListOf<String>()
        lines += "Recent trajectory (oldest -> newest):"
        for (item in history) {
            val status = if (item.success) "ok" else "fail"
            var line = "- step=${item.step}, action=${item.action}, result=$status"
            if (item.error.isNotBlank()) {
                line += ", error=${item.error.take(120)}"
            }
            lines += line
        }
        return lines.joinToString("\n")
    }
}
