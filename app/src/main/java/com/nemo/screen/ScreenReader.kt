package com.nemo.screen

import android.graphics.Rect
import android.view.accessibility.AccessibilityNodeInfo

/**
 * Screen understanding via AccessibilityService.
 * Converts AccessibilityNodeInfo tree → structured ScreenState.
 * Replaces Python's XML parser (parser.py) with zero-latency native access.
 */

data class UIElement(
    val index: Int,
    val resourceId: String = "",
    val className: String = "",
    val text: String = "",
    val contentDesc: String = "",
    val bounds: Rect = Rect(),
    val clickable: Boolean = false,
    val scrollable: Boolean = false,
    val editable: Boolean = false,
    val checked: Boolean? = null,
    val enabled: Boolean = true,
) {
    val center: Pair<Int, Int>
        get() = (bounds.left + bounds.right) / 2 to (bounds.top + bounds.bottom) / 2

    val displayText: String
        get() = text.ifBlank { contentDesc.ifBlank { resourceId.substringAfterLast("/", className) } }

    fun toPromptStr(): String {
        val parts = mutableListOf("[$index]")
        if (text.isNotBlank()) parts.add("\"$text\"")
        if (contentDesc.isNotBlank() && contentDesc != text) parts.add("($contentDesc)")
        parts.add(className.substringAfterLast("."))
        val attrs = mutableListOf<String>()
        if (clickable) attrs.add("clickable")
        if (scrollable) attrs.add("scrollable")
        if (editable) attrs.add("editable")
        if (attrs.isNotEmpty()) parts.add("[${attrs.joinToString(",")}]")
        return parts.joinToString(" ")
    }
}

data class ScreenState(
    val activity: String = "",
    val packageName: String = "",
    val elements: List<UIElement> = emptyList(),
    val timestamp: Long = System.currentTimeMillis(),
) {
    val interactiveElements: List<UIElement>
        get() = elements.filter { it.clickable || it.scrollable || it.editable }

    fun toPromptStr(): String = buildString {
        appendLine("Screen: $activity")
        appendLine("Interactive elements (${interactiveElements.size}):")
        interactiveElements.forEach { appendLine("  ${it.toPromptStr()}") }
    }

    fun computeHash(): String {
        val content = elements.joinToString("|") { "${it.className}:${it.text}:${it.bounds}" }
        return content.hashCode().toUInt().toString(16).padStart(8, '0')
    }
}

object ScreenReader {

    /**
     * Parse AccessibilityNodeInfo tree into ScreenState.
     * This is the core replacement for Python's parse_ui_hierarchy().
     * Zero latency — reads directly from the accessibility framework.
     */
    fun parse(rootNode: AccessibilityNodeInfo?, packageName: String = ""): ScreenState {
        if (rootNode == null) return ScreenState()

        val elements = mutableListOf<UIElement>()
        var idx = 0

        fun walk(node: AccessibilityNodeInfo) {
            val bounds = Rect()
            node.getBoundsInScreen(bounds)

            val element = UIElement(
                index = idx++,
                resourceId = node.viewIdResourceName.orEmpty(),
                className = node.className?.toString().orEmpty(),
                text = node.text?.toString().orEmpty(),
                contentDesc = node.contentDescription?.toString().orEmpty(),
                bounds = bounds,
                clickable = node.isClickable,
                scrollable = node.isScrollable,
                editable = node.isEditable,
                checked = if (node.isCheckable) node.isChecked else null,
                enabled = node.isEnabled,
            )
            elements.add(element)

            for (i in 0 until node.childCount) {
                node.getChild(i)?.let { child ->
                    walk(child)
                    child.recycle()
                }
            }
        }

        walk(rootNode)

        return ScreenState(
            activity = packageName,
            packageName = packageName,
            elements = elements,
        )
    }
}
