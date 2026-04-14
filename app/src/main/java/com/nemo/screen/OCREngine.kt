package com.nemo.screen

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.TextRecognizer
import com.google.mlkit.vision.text.chinese.ChineseTextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * ML Kit-based OCR engine optimized for Chinese text on Android screens.
 *
 * Replaces Python's PaddleOCR — uses offline Chinese text recognition
 * model from Google ML Kit for zero-latency on-device OCR.
 */
class OCREngine {

    /** A recognized text region on screen. */
    data class TextBlock(
        val text: String,
        val confidence: Float,
        val bounds: IntArray // [left, top, right, bottom]
    ) {
        val centerX: Int get() = (bounds[0] + bounds[2]) / 2
        val centerY: Int get() = (bounds[1] + bounds[3]) / 2

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is TextBlock) return false
            return text == other.text && confidence == other.confidence && bounds.contentEquals(other.bounds)
        }

        override fun hashCode(): Int = text.hashCode() * 31 + confidence.hashCode()
    }

    /** Full OCR result for one screenshot. */
    data class OCRResult(
        val blocks: List<TextBlock>,
        val fullText: String,
        val elapsedMs: Long
    ) {
        /** Find all blocks containing the query text (case-insensitive). */
        fun findText(query: String): List<TextBlock> {
            val q = query.lowercase()
            return blocks.filter { q in it.text.lowercase() }
        }
    }

    private val recognizer: TextRecognizer = TextRecognition.getClient(
        ChineseTextRecognizerOptions.Builder().build()
    )

    /**
     * Run OCR on a screenshot bitmap.
     * Returns structured text blocks with confidence and positions.
     */
    suspend fun recognize(bitmap: Bitmap): OCRResult {
        val startTime = System.currentTimeMillis()
        val image = InputImage.fromBitmap(bitmap, 0)

        val visionText = suspendCancellableCoroutine { cont ->
            recognizer.process(image)
                .addOnSuccessListener { text -> cont.resume(text) }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }

        val blocks = mutableListOf<TextBlock>()

        for (textBlock in visionText.textBlocks) {
            for (line in textBlock.lines) {
                for (element in line.elements) {
                    val box = element.boundingBox ?: continue
                    val text = element.text.trim()
                    if (text.isEmpty()) continue

                    // ML Kit confidence: element-level confidence (0.0 to 1.0)
                    val confidence = element.confidence ?: 0.8f

                    blocks.add(
                        TextBlock(
                            text = text,
                            confidence = confidence,
                            bounds = intArrayOf(box.left, box.top, box.right, box.bottom)
                        )
                    )
                }
            }
        }

        val elapsed = System.currentTimeMillis() - startTime

        return OCRResult(
            blocks = blocks.sortedWith(compareBy({ it.bounds[1] }, { it.bounds[0] })),
            fullText = visionText.text,
            elapsedMs = elapsed
        )
    }

    /**
     * Run OCR and filter results by minimum confidence.
     */
    suspend fun recognizeFiltered(
        bitmap: Bitmap,
        minConfidence: Float = 0.3f
    ): OCRResult {
        val result = recognize(bitmap)
        val filtered = result.blocks.filter { it.confidence >= minConfidence }
        return result.copy(blocks = filtered)
    }

    /**
     * Convert OCR results to SignalCandidates for DempsterShafer fusion.
     */
    fun toSignalCandidates(
        blocks: List<TextBlock>,
        minConfidence: Double = 0.4
    ): List<SignalCandidate> = candidatesFromOcr(blocks, minConfidence)

    /** Release ML Kit resources. */
    fun close() {
        recognizer.close()
    }
}
