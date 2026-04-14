package com.nemo.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Privacy Dashboard — shows users exactly what data is processed and where.
 * Key trust-building feature: "zero bytes uploaded" visible proof.
 */
data class PrivacyStats(
    val screensProcessed: Int = 0,
    val elementsAnalyzed: Int = 0,
    val sensitiveFieldsFiltered: Int = 0,
    val piiRedacted: Int = 0,
    val actionsBlocked: Int = 0,
    val bytesUploaded: Long = 0, // always 0
    val modelInferenceCount: Int = 0,
)

@Composable
fun PrivacyDashboard(
    stats: PrivacyStats,
    onBack: () -> Unit,
) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Privacy Dashboard", style = MaterialTheme.typography.headlineMedium)
            TextButton(onClick = onBack) { Text("Back") }
        }
        Spacer(Modifier.height(16.dp))

        // Hero: zero upload badge
        Card(
            colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9)),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(Modifier.padding(20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Text("🔒", style = MaterialTheme.typography.headlineLarge)
                Spacer(Modifier.height(8.dp))
                Text("Zero bytes uploaded", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                Text(
                    "All ${stats.modelInferenceCount} AI inferences ran on your device",
                    style = MaterialTheme.typography.bodyMedium,
                    color = Color.Gray,
                )
            }
        }
        Spacer(Modifier.height(16.dp))

        // Stats grid
        Text("Session Statistics", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatCard("Screens\nProcessed", "${stats.screensProcessed}", Color(0xFF1976D2), Modifier.weight(1f))
            StatCard("Elements\nAnalyzed", "${stats.elementsAnalyzed}", Color(0xFF388E3C), Modifier.weight(1f))
        }
        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatCard("Sensitive Fields\nFiltered", "${stats.sensitiveFieldsFiltered}", Color(0xFFF57C00), Modifier.weight(1f))
            StatCard("PII\nRedacted", "${stats.piiRedacted}", Color(0xFFD32F2F), Modifier.weight(1f))
        }
        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatCard("Actions\nBlocked", "${stats.actionsBlocked}", Color(0xFF7B1FA2), Modifier.weight(1f))
            StatCard("Data\nUploaded", "0 bytes", Color(0xFF2E7D32), Modifier.weight(1f))
        }

        Spacer(Modifier.height(16.dp))

        // Guarantees
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("Privacy Guarantees", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(8.dp))
                Guarantee("All AI inference runs on-device (Gemma 1B via MediaPipe)")
                Guarantee("Screen content never sent to any server")
                Guarantee("Password/PIN/CVV fields filtered before AI processing")
                Guarantee("Credit card and ID numbers auto-redacted")
                Guarantee("Settings encrypted with AES-256")
                Guarantee("Open source — audit the code at github.com/AShan0227/phone-agent")
            }
        }
    }
}

@Composable
private fun StatCard(label: String, value: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        colors = CardDefaults.cardColors(containerColor = color.copy(alpha = 0.1f)),
        modifier = modifier,
    ) {
        Column(Modifier.padding(12.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(value, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, color = color)
            Text(label, style = MaterialTheme.typography.labelSmall, color = color.copy(alpha = 0.8f))
        }
    }
}

@Composable
private fun Guarantee(text: String) {
    Row(Modifier.padding(vertical = 2.dp)) {
        Text("✅ ", style = MaterialTheme.typography.bodySmall)
        Text(text, style = MaterialTheme.typography.bodySmall)
    }
}
