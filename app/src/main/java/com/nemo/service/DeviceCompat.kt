package com.nemo.service

import android.os.Build
import android.util.DisplayMetrics
import com.nemo.screen.ScreenState

/**
 * Device compatibility utilities for real-device adaptation.
 * Checks AccessibilityService capabilities and screen info.
 */
data class DeviceInfo(
    val manufacturer: String = Build.MANUFACTURER,
    val model: String = Build.MODEL,
    val sdkVersion: Int = Build.VERSION.SDK_INT,
    val screenWidth: Int = 0,
    val screenHeight: Int = 0,
    val density: Float = 0f,
)

data class CompatCheck(
    val name: String,
    val passed: Boolean,
    val message: String,
)

object DeviceCompat {

    fun getDeviceInfo(service: NemoAccessibilityService): DeviceInfo {
        val dm = service.resources.displayMetrics
        return DeviceInfo(
            screenWidth = dm.widthPixels,
            screenHeight = dm.heightPixels,
            density = dm.density,
        )
    }

    /**
     * Run non-destructive compatibility checks.
     */
    fun runChecks(service: NemoAccessibilityService): List<CompatCheck> {
        val checks = mutableListOf<CompatCheck>()

        // 1. SDK version
        checks.add(CompatCheck(
            "sdk_version",
            Build.VERSION.SDK_INT >= 28,
            "SDK ${Build.VERSION.SDK_INT} (min: 28)",
        ))

        // 2. Screen size readable
        val (w, h) = service.getScreenSize()
        checks.add(CompatCheck(
            "screen_size",
            w > 0 && h > 0,
            "${w}x${h}",
        ))

        // 3. Accessibility tree available
        val screen = service.getScreenState()
        checks.add(CompatCheck(
            "accessibility_tree",
            screen.elements.isNotEmpty(),
            "${screen.elements.size} elements found",
        ))

        // 4. Interactive elements present
        val interactive = screen.interactiveElements
        checks.add(CompatCheck(
            "interactive_elements",
            interactive.isNotEmpty(),
            "${interactive.size} interactive elements",
        ))

        // 5. Gesture support
        checks.add(CompatCheck(
            "gesture_support",
            Build.VERSION.SDK_INT >= 24,  // GestureDescription added in API 24
            "API ${Build.VERSION.SDK_INT} (gestures need API 24+)",
        ))

        return checks
    }

    /**
     * Analyze screen quality for a specific app.
     */
    fun analyzeAppCompat(screen: ScreenState): Map<String, Any> {
        val elements = screen.elements
        val interactive = screen.interactiveElements
        val hasWebView = elements.any { "WebView" in it.className }
        val hasRecyclerView = elements.any { "RecyclerView" in it.className }
        val textElements = elements.filter { it.text.isNotBlank() }

        return mapOf(
            "total_elements" to elements.size,
            "interactive_elements" to interactive.size,
            "text_elements" to textElements.size,
            "has_webview" to hasWebView,
            "has_recyclerview" to hasRecyclerView,
            "coverage_ratio" to if (elements.isNotEmpty()) interactive.size.toFloat() / elements.size else 0f,
            "quality" to when {
                interactive.size >= 5 && textElements.size >= 3 -> "good"
                interactive.size >= 2 -> "fair"
                else -> "poor"
            },
        )
    }
}
