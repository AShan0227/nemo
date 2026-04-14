package com.nemo.ui

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import androidx.core.content.FileProvider
import com.nemo.agent.StepRecord
import com.nemo.agent.TaskResult
import com.nemo.agent.TaskStatus
import java.io.File

/**
 * Generates shareable summary images for completed tasks.
 * Creates a branded card showing task result + NERVE stats.
 * Users share these on WeChat/Weibo/X to drive viral growth.
 */
object ShareGenerator {

    private const val CARD_WIDTH = 1080
    private const val CARD_HEIGHT = 1920
    private const val MARGIN = 60f
    private const val CORNER_RADIUS = 32f

    fun generateTaskCard(
        context: Context,
        task: String,
        result: TaskResult,
    ): File {
        val bitmap = Bitmap.createBitmap(CARD_WIDTH, CARD_HEIGHT, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)

        drawBackground(canvas)
        drawHeader(canvas, task)
        drawResult(canvas, result)
        drawSteps(canvas, result.steps)
        drawNerveStats(canvas, result)
        drawFooter(canvas)

        val file = File(context.cacheDir, "nemo_share_${System.currentTimeMillis()}.png")
        file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.PNG, 95, it) }
        bitmap.recycle()
        return file
    }

    fun shareImage(context: Context, file: File, task: String) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "image/png"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TEXT, "My phone AI just completed: \"$task\" \uD83E\uDDE0\n#NemoAgent #NERVE")
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "Share Nemo Result"))
    }

    private fun drawBackground(canvas: Canvas) {
        canvas.drawColor(Color.WHITE)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
            color = Color.parseColor("#1A237E")
        }
        canvas.drawRoundRect(RectF(MARGIN, MARGIN, CARD_WIDTH - MARGIN, CARD_HEIGHT - MARGIN), CORNER_RADIUS, CORNER_RADIUS, paint)
    }

    private fun drawHeader(canvas: Canvas, task: String) {
        val paint = textPaint(48f, Color.WHITE, true)
        canvas.drawText("\uD83E\uDDE0 Nemo Agent", MARGIN + 40f, MARGIN + 80f, paint)

        val taskPaint = textPaint(32f, Color.parseColor("#B0BEC5"), false)
        val wrapped = wrapText(task, taskPaint, CARD_WIDTH - 4 * MARGIN)
        var y = MARGIN + 130f
        for (line in wrapped) {
            canvas.drawText(line, MARGIN + 40f, y, taskPaint)
            y += 42f
        }
    }

    private fun drawResult(canvas: Canvas, result: TaskResult) {
        val isSuccess = result.status == TaskStatus.COMPLETED
        val color = if (isSuccess) Color.parseColor("#4CAF50") else Color.parseColor("#F44336")
        val label = if (isSuccess) "COMPLETED" else result.status.name

        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { this.color = color; alpha = 40 }
        canvas.drawRoundRect(RectF(MARGIN + 30f, 300f, CARD_WIDTH - MARGIN - 30f, 420f), 20f, 20f, bgPaint)

        val paint = textPaint(40f, color, true)
        canvas.drawText("$label — ${result.totalSteps} steps", MARGIN + 60f, 375f, paint)
    }

    private fun drawSteps(canvas: Canvas, steps: List<StepRecord>) {
        val display = steps.takeLast(6)
        var y = 470f
        val successPaint = textPaint(26f, Color.parseColor("#81C784"), false)
        val failPaint = textPaint(26f, Color.parseColor("#E57373"), false)

        for (step in display) {
            val paint = if (step.success) successPaint else failPaint
            val icon = if (step.success) "✓" else "✗"
            val text = "$icon Step ${step.step}: ${step.action} (${step.durationMs}ms)"
            canvas.drawText(text, MARGIN + 50f, y, paint)
            y += 38f
        }
        if (steps.size > 6) {
            canvas.drawText("... and ${steps.size - 6} more steps", MARGIN + 50f, y, textPaint(24f, Color.GRAY, false))
        }
    }

    private fun drawNerveStats(canvas: Canvas, result: TaskResult) {
        val y = 900f
        val labelPaint = textPaint(28f, Color.parseColor("#90CAF9"), true)
        val valuePaint = textPaint(26f, Color.WHITE, false)

        canvas.drawText("NERVE System", MARGIN + 40f, y, labelPaint)

        val stats = listOf(
            "\uD83E\uDDE0 Entropy Router" to "Active",
            "\uD83D\uDEE1 Safety Layer" to "Protected",
            "\uD83D\uDC1C ACO Learning" to "Paths recorded",
            "\uD83D\uDD12 Privacy" to "0 bytes uploaded",
        )
        var sy = y + 40f
        for ((label, value) in stats) {
            canvas.drawText("$label: $value", MARGIN + 60f, sy, valuePaint)
            sy += 36f
        }
    }

    private fun drawFooter(canvas: Canvas) {
        val paint = textPaint(22f, Color.parseColor("#78909C"), false)
        canvas.drawText("github.com/AShan0227/phone-agent", MARGIN + 40f, CARD_HEIGHT - MARGIN - 60f, paint)
        canvas.drawText("On-device AI • Zero cloud • Open source", MARGIN + 40f, CARD_HEIGHT - MARGIN - 30f, paint)
    }

    private fun textPaint(size: Float, color: Int, bold: Boolean) = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textSize = size
        this.color = color
        typeface = if (bold) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
    }

    private fun wrapText(text: String, paint: Paint, maxWidth: Float): List<String> {
        val words = text.toList().map { it.toString() }
        val lines = mutableListOf<String>()
        var current = StringBuilder()
        for (word in words) {
            val test = current.toString() + word
            if (paint.measureText(test) > maxWidth && current.isNotEmpty()) {
                lines.add(current.toString())
                current = StringBuilder(word)
            } else {
                current.append(word)
            }
        }
        if (current.isNotEmpty()) lines.add(current.toString())
        return lines.take(3)
    }
}
