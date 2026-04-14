package com.nemo.ui

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Context
import android.content.Intent
import android.provider.Settings
import android.view.accessibility.AccessibilityManager
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.nemo.service.NemoAccessibilityService

/**
 * Accessibility permission detection + guided setup.
 * Shows step-by-step instructions to enable NemoAccessibilityService.
 */
object AccessibilityHelper {

    fun isServiceEnabled(context: Context): Boolean {
        val am = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as? AccessibilityManager
            ?: return false
        val enabled = am.getEnabledAccessibilityServiceList(AccessibilityServiceInfo.FEEDBACK_ALL_MASK)
        return enabled.any { it.resolveInfo.serviceInfo.name == NemoAccessibilityService::class.java.name }
    }

    fun openAccessibilitySettings(context: Context) {
        context.startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
            addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        })
    }
}

@Composable
fun AccessibilityGuideCard(
    onOpenSettings: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    var isEnabled by remember { mutableStateOf(AccessibilityHelper.isServiceEnabled(context)) }

    // Re-check every time the composable recomposes (e.g. on resume)
    LaunchedEffect(Unit) {
        isEnabled = AccessibilityHelper.isServiceEnabled(context)
    }

    if (isEnabled) return

    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer),
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(
                "Accessibility Service Required",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )
            Spacer(Modifier.height(8.dp))
            Text("Nemo needs Accessibility Service to read screen content and perform actions.")
            Spacer(Modifier.height(8.dp))

            Text("Setup steps:", style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.height(4.dp))
            val steps = listOf(
                "1. Tap 'Open Settings' below",
                "2. Find 'Nemo Agent' in the list",
                "3. Toggle the switch ON",
                "4. Confirm 'Allow' in the dialog",
                "5. Return to this app",
            )
            steps.forEach { Text(it, style = MaterialTheme.typography.bodySmall) }

            Spacer(Modifier.height(12.dp))
            Button(
                onClick = onOpenSettings,
                modifier = Modifier.align(Alignment.End),
            ) {
                Text("Open Accessibility Settings")
            }
        }
    }
}
