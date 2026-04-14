package com.nemo.agent

import android.view.accessibility.AccessibilityNodeInfo
import com.nemo.screen.ScreenState
import com.nemo.service.NemoAccessibilityService
import kotlinx.coroutines.delay
import kotlinx.serialization.json.*

/**
 * Action builder and executor.
 * Decouples LLM JSON decision parsing from device operation execution.
 */
enum class BuiltActionType {
    TAP,
    TYPE_TEXT,
    SCROLL,
    BACK,
    HOME,
    LAUNCH,
    WAIT,
}

data class BuiltAction(
    val type: BuiltActionType,
    val params: Map<String, Any> = emptyMap(),
    val description: String = "",
)

class ActionBuilder {
    private val json = Json { ignoreUnknownKeys = true }

    data class Decision(
        val reasoning: String,
        val actionName: String,
        val params: Map<String, Any>,
    )

    fun parseDecision(raw: String): Decision {
        val start = raw.indexOf('{')
        val end = raw.lastIndexOf('}') + 1
        if (start < 0 || end <= start) {
            throw IllegalArgumentException("No JSON found in: ${raw.take(200)}")
        }
        val jsonObject = json.parseToJsonElement(raw.substring(start, end)).jsonObject
        val reasoning = jsonObject["reasoning"]?.jsonPrimitive?.contentOrNull.orEmpty()
        val action = jsonObject["action"]?.jsonPrimitive?.contentOrNull.orEmpty()
        val params = parseParams(jsonObject["params"]?.jsonObject)
        return Decision(reasoning = reasoning, actionName = action, params = params)
    }

    fun buildAction(name: String, params: Map<String, Any>, screen: ScreenState): BuiltAction? {
        val elements = screen.interactiveElements

        fun resolveCenter(index: Int): Pair<Int, Int>? {
            return if (index in elements.indices) elements[index].center else null
        }

        return when (name) {
            "tap" -> {
                val idx = params["index"].toIntSafe(default = 0)
                val center = resolveCenter(idx) ?: return null
                BuiltAction(
                    type = BuiltActionType.TAP,
                    params = mapOf("x" to center.first, "y" to center.second, "index" to idx),
                    description = "Tap [$idx]",
                )
            }

            "type_text" -> {
                val idx = params["index"].toIntSafe(default = 0)
                val text = params["text"]?.toString().orEmpty()
                val center = resolveCenter(idx) ?: return null
                BuiltAction(
                    type = BuiltActionType.TYPE_TEXT,
                    params = mapOf(
                        "x" to center.first,
                        "y" to center.second,
                        "index" to idx,
                        "text" to text,
                    ),
                    description = "Type '$text' at [$idx]",
                )
            }

            "scroll" -> {
                BuiltAction(
                    type = BuiltActionType.SCROLL,
                    params = mapOf(
                        "direction" to (params["direction"]?.toString() ?: "down"),
                        "amount" to params["amount"].toIntSafe(default = 1),
                    ),
                    description = "Scroll",
                )
            }

            "back" -> BuiltAction(BuiltActionType.BACK, description = "Back")
            "home" -> BuiltAction(BuiltActionType.HOME, description = "Home")

            "launch" -> {
                val pkg = params["package"]?.toString().orEmpty()
                if (pkg.isBlank()) return null
                BuiltAction(
                    type = BuiltActionType.LAUNCH,
                    params = mapOf("package" to pkg),
                    description = "Launch $pkg",
                )
            }

            "wait" -> {
                val ms = params["ms"].toLongSafe(default = 1000L)
                BuiltAction(
                    type = BuiltActionType.WAIT,
                    params = mapOf("ms" to ms),
                    description = "Wait ${ms}ms",
                )
            }

            else -> null
        }
    }

    suspend fun executeAction(service: NemoAccessibilityService, action: BuiltAction): Boolean {
        return when (action.type) {
            BuiltActionType.TAP -> {
                val x = action.params["x"].toIntSafe(default = 0)
                val y = action.params["y"].toIntSafe(default = 0)
                service.tap(x, y)
            }

            BuiltActionType.TYPE_TEXT -> {
                val x = action.params["x"].toIntSafe(default = 0)
                val y = action.params["y"].toIntSafe(default = 0)
                val text = action.params["text"]?.toString().orEmpty()
                val tapped = service.tap(x, y)
                if (!tapped) return false
                delay(300)

                val root = service.rootInActiveWindow
                val nodes = root?.let { findEditableNodes(it) }.orEmpty()
                val typed = nodes.firstOrNull()?.let { service.inputText(it, text) } ?: false
                root?.recycle()
                typed
            }

            BuiltActionType.SCROLL -> {
                val direction = action.params["direction"]?.toString() ?: "down"
                val amount = action.params["amount"].toIntSafe(default = 1).coerceAtLeast(1)
                val (w, h) = service.getScreenSize()
                val cx = w / 2
                val cy = h / 2
                val dy = (h / 4) * amount
                if (direction == "down") {
                    service.swipe(cx, cy, cx, cy - dy)
                } else {
                    service.swipe(cx, cy, cx, cy + dy)
                }
            }

            BuiltActionType.BACK -> service.pressBack()
            BuiltActionType.HOME -> service.pressHome()
            BuiltActionType.LAUNCH -> {
                val pkg = action.params["package"]?.toString().orEmpty()
                service.launchApp(pkg)
            }

            BuiltActionType.WAIT -> {
                val ms = action.params["ms"].toLongSafe(default = 1000L)
                delay(ms)
                true
            }
        }
    }

    private fun parseParams(jsonObject: JsonObject?): Map<String, Any> {
        if (jsonObject == null) return emptyMap()
        return jsonObject.mapValues { (_, value) -> value.toAny() }
    }

    private fun JsonElement.toAny(): Any {
        return when (this) {
            is JsonObject -> this.mapValues { (_, value) -> value.toAny() }
            is JsonArray -> this.map { element -> element.toAny() }
            is JsonNull -> ""
            is JsonPrimitive -> {
                when {
                    isString -> content
                    booleanOrNull != null -> boolean
                    longOrNull != null -> long
                    doubleOrNull != null -> double
                    else -> content
                }
            }
        }
    }

    private fun findEditableNodes(node: AccessibilityNodeInfo): List<AccessibilityNodeInfo> {
        val result = mutableListOf<AccessibilityNodeInfo>()
        if (node.isEditable) result += node
        for (i in 0 until node.childCount) {
            node.getChild(i)?.let { child ->
                result += findEditableNodes(child)
            }
        }
        return result
    }

    private fun Any?.toIntSafe(default: Int): Int {
        return when (this) {
            is Number -> this.toInt()
            is String -> this.toIntOrNull() ?: default
            else -> default
        }
    }

    private fun Any?.toLongSafe(default: Long): Long {
        return when (this) {
            is Number -> this.toLong()
            is String -> this.toLongOrNull() ?: default
            else -> default
        }
    }
}
