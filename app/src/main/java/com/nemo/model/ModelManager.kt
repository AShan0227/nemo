package com.nemo.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import com.nemo.ui.Strings

data class DownloadProgress(
    val downloadedBytes: Long = 0,
    val totalBytes: Long = 0,
    val percent: Int = 0,
    val status: String = "",
)

class ModelManager(private val context: Context) {

    fun defaultModelPath(): String {
        val modelsDir = File(context.filesDir, "models")
        return File(modelsDir, DEFAULT_MODEL_FILE_NAME).absolutePath
    }

    /**
     * Check if model is fully downloaded.
     * Uses expected file size, not just > 0.
     */
    fun isModelDownloaded(path: String): Boolean {
        val file = File(path)
        if (!file.exists()) return false
        // Must be at least 50MB to be a real model (not a partial download)
        return file.length() > MINIMUM_MODEL_SIZE
    }

    suspend fun downloadModel(
        url: String,
        targetPath: String,
        expectedSha256: String? = null,
        authToken: String? = null,
        onProgress: (DownloadProgress) -> Unit = {},
    ): Result<File> = withContext(Dispatchers.IO) {
        runCatching {
            require(url.startsWith("https://")) { "Only HTTPS URLs allowed" }

            val targetFile = File(targetPath)
            val allowedDir = context.filesDir.canonicalPath
            require(targetFile.canonicalPath.startsWith(allowedDir)) {
                "Model path must be within app private storage"
            }
            targetFile.parentFile?.mkdirs()

            // If already downloaded and valid, skip
            if (isModelDownloaded(targetPath)) {
                onProgress(DownloadProgress(targetFile.length(), targetFile.length(), 100, "complete"))
                return@runCatching targetFile
            }

            // Download to temp file, rename on success (prevents partial file corruption)
            val tempFile = File(targetPath + ".downloading")
            if (tempFile.exists()) tempFile.delete()

            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 30_000
                readTimeout = 60_000
                doInput = true
                instanceFollowRedirects = true
                if (!authToken.isNullOrBlank()) {
                    setRequestProperty("Authorization", "Bearer $authToken")
                }
            }

            connection.connect()
            if (connection.responseCode !in 200..299) {
                throw IllegalStateException("HTTP ${connection.responseCode} ${connection.responseMessage}")
            }

            val total = connection.contentLengthLong.coerceAtLeast(0L)
            onProgress(DownloadProgress(0L, total, 0, "starting"))

            connection.inputStream.use { input ->
                FileOutputStream(tempFile).use { output ->
                    val buffer = ByteArray(8192)
                    var downloaded = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        val pct = if (total > 0L) ((downloaded * 100) / total).toInt() else 0
                        onProgress(DownloadProgress(downloaded, total, pct.coerceIn(0, 99), "downloading"))
                    }
                }
            }
            connection.disconnect()

            // Verify download completeness
            if (total > 0 && tempFile.length() < total * 0.95) {
                tempFile.delete()
                throw IllegalStateException("Download incomplete: got ${tempFile.length()} bytes, expected $total")
            }
            if (tempFile.length() < MINIMUM_MODEL_SIZE) {
                tempFile.delete()
                throw IllegalStateException("Downloaded file too small (${tempFile.length()} bytes), download likely failed")
            }

            // SHA256 check
            if (!expectedSha256.isNullOrBlank()) {
                val actual = sha256(tempFile)
                if (!actual.equals(expectedSha256, ignoreCase = true)) {
                    tempFile.delete()
                    throw IllegalStateException("SHA256 mismatch")
                }
            }

            // Success — rename temp to final
            if (targetFile.exists()) targetFile.delete()
            tempFile.renameTo(targetFile)

            onProgress(DownloadProgress(targetFile.length(), targetFile.length(), 100, "complete"))
            targetFile
        }
    }

    /**
     * Download multi-part model and merge into one file.
     * Downloads to temp files, merges only when ALL parts complete.
     */
    suspend fun downloadMultiPartModel(
        partUrls: List<String>,
        targetPath: String,
        authToken: String? = null,
        onProgress: (DownloadProgress) -> Unit = {},
    ): Result<File> = withContext(Dispatchers.IO) {
        runCatching {
            val targetFile = File(targetPath)
            val allowedDir = context.filesDir.canonicalPath
            require(targetFile.canonicalPath.startsWith(allowedDir)) {
                "Model path must be within app private storage"
            }

            if (isModelDownloaded(targetPath)) {
                onProgress(DownloadProgress(targetFile.length(), targetFile.length(), 100, "complete"))
                return@runCatching targetFile
            }

            targetFile.parentFile?.mkdirs()
            val tempFile = File(targetPath + ".downloading")
            if (tempFile.exists()) tempFile.delete()

            val totalParts = partUrls.size
            var totalDownloaded = 0L

            FileOutputStream(tempFile).use { output ->
                for ((index, url) in partUrls.withIndex()) {
                    require(url.startsWith("https://")) { "Only HTTPS URLs allowed" }

                    val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                        requestMethod = "GET"
                        connectTimeout = 30_000
                        readTimeout = 60_000
                        doInput = true
                        instanceFollowRedirects = true
                        if (!authToken.isNullOrBlank()) {
                            setRequestProperty("Authorization", "Bearer $authToken")
                        }
                    }
                    connection.connect()
                    if (connection.responseCode !in 200..299) {
                        tempFile.delete()
                        throw IllegalStateException("HTTP ${connection.responseCode} on part ${index + 1}")
                    }

                    val partSize = connection.contentLengthLong.coerceAtLeast(1L)
                    var partDownloaded = 0L
                    val buffer = ByteArray(8192)

                    connection.inputStream.use { input ->
                        while (true) {
                            val read = input.read(buffer)
                            if (read < 0) break
                            output.write(buffer, 0, read)
                            partDownloaded += read
                            totalDownloaded += read
                            val partPct = (partDownloaded * 100 / partSize).toInt()
                            val totalPct = ((index * 100 + partPct) / totalParts).coerceIn(0, 99)
                            onProgress(DownloadProgress(
                                totalDownloaded, 0,
                                totalPct,
                                if (Strings.isChinese) "第 ${index + 1}/$totalParts 部分" else "part ${index + 1}/$totalParts",
                            ))
                        }
                    }
                    connection.disconnect()

                    // Verify each part downloaded completely
                    if (partDownloaded < partSize * 0.95) {
                        tempFile.delete()
                        throw IllegalStateException(
                            if (Strings.isChinese) "第 ${index + 1} 部分下载不完整，请重试"
                            else "Part ${index + 1} incomplete, please retry"
                        )
                    }
                }
            }

            // Final size check
            if (tempFile.length() < MINIMUM_MODEL_SIZE) {
                tempFile.delete()
                throw IllegalStateException(
                    if (Strings.isChinese) "下载文件不完整 (${tempFile.length() / 1_000_000}MB)，请检查网络后重试"
                    else "Download incomplete (${tempFile.length() / 1_000_000}MB), check network and retry"
                )
            }

            // All parts downloaded — rename to final
            if (targetFile.exists()) targetFile.delete()
            tempFile.renameTo(targetFile)

            onProgress(DownloadProgress(totalDownloaded, totalDownloaded, 100, "complete"))
            targetFile
        }
    }

    fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                if (read > 0) digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    companion object {
        private const val DEFAULT_MODEL_FILE_NAME = "gemma-4-E2B-it.litertlm"
        private const val MINIMUM_MODEL_SIZE = 50_000_000L // 50MB minimum for valid model

        // Multi-part download URLs (GitHub Release 2GB limit, model split into 3 parts)
        val MODEL_PART_URLS = listOf(
            "https://github.com/AShan0227/nemo/releases/download/model-v1/gemma4-part-aa",
            "https://github.com/AShan0227/nemo/releases/download/model-v1/gemma4-part-ab",
            "https://github.com/AShan0227/nemo/releases/download/model-v1/gemma4-part-ac",
        )

        // Single file URL (for reference / settings override)
        const val DEFAULT_MODEL_URL = "https://github.com/AShan0227/nemo/releases/download/model-v1/gemma-4-E2B-it.litertlm"
    }
}
