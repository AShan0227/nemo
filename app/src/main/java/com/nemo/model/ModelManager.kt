package com.nemo.model

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Model download and validation manager.
 */
data class DownloadProgress(
    val downloadedBytes: Long,
    val totalBytes: Long,
    val percent: Int,
    val status: String,
)

class ModelManager(private val context: Context) {

    fun defaultModelPath(): String {
        val modelsDir = File(context.filesDir, "models")
        return File(modelsDir, DEFAULT_MODEL_FILE_NAME).absolutePath
    }

    fun isModelDownloaded(path: String): Boolean {
        val file = File(path)
        return file.exists() && file.length() > 0
    }

    suspend fun downloadModel(
        url: String,
        targetPath: String,
        expectedSha256: String? = null,
        authToken: String? = null,
        onProgress: (DownloadProgress) -> Unit = {},
    ): Result<File> = withContext(Dispatchers.IO) {
        runCatching {
            require(url.startsWith("https://")) { "Only HTTPS URLs allowed for model download" }

            val targetFile = File(targetPath)
            val allowedDir = context.filesDir.canonicalPath
            require(targetFile.canonicalPath.startsWith(allowedDir)) {
                "Model path must be within app private storage"
            }
            targetFile.parentFile?.mkdirs()

            if (targetFile.exists() && targetFile.length() > 0) {
                if (expectedSha256.isNullOrBlank()) {
                    onProgress(
                        DownloadProgress(
                            downloadedBytes = targetFile.length(),
                            totalBytes = targetFile.length(),
                            percent = 100,
                            status = "already_downloaded",
                        )
                    )
                    return@runCatching targetFile
                }
                val currentHash = sha256(targetFile)
                if (currentHash.equals(expectedSha256, ignoreCase = true)) {
                    onProgress(
                        DownloadProgress(
                            downloadedBytes = targetFile.length(),
                            totalBytes = targetFile.length(),
                            percent = 100,
                            status = "already_verified",
                        )
                    )
                    return@runCatching targetFile
                }
                targetFile.delete()
            }

            val connection = (URL(url).openConnection() as HttpURLConnection).apply {
                requestMethod = "GET"
                connectTimeout = 30_000
                readTimeout = 30_000
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
                FileOutputStream(targetFile).use { output ->
                    val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                    var downloaded = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count

                        val pct = if (total > 0L) ((downloaded * 100) / total).toInt() else 0
                        onProgress(
                            DownloadProgress(
                                downloadedBytes = downloaded,
                                totalBytes = total,
                                percent = pct.coerceIn(0, 100),
                                status = "downloading",
                            )
                        )
                    }
                }
            }

            if (!expectedSha256.isNullOrBlank()) {
                val actual = sha256(targetFile)
                if (!actual.equals(expectedSha256, ignoreCase = true)) {
                    targetFile.delete()
                    throw IllegalStateException("SHA256 mismatch: expected=$expectedSha256 actual=$actual")
                }
            }

            onProgress(
                DownloadProgress(
                    downloadedBytes = targetFile.length(),
                    totalBytes = targetFile.length(),
                    percent = 100,
                    status = "completed",
                )
            )
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
        private const val DEFAULT_MODEL_FILE_NAME = "gemma3-1b-it-int4.task"
        const val DEFAULT_MODEL_URL = "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task"
    }
}
