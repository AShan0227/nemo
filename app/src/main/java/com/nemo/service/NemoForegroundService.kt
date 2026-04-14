package com.nemo.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import com.nemo.NemoApp
import com.nemo.agent.PhoneAgent
import com.nemo.agent.TaskResult
import com.nemo.agent.TaskStatus
import com.nemo.ui.MainActivity
import kotlinx.coroutines.*

/**
 * Foreground service for background task execution.
 * Agent continues running even if user closes the app.
 * Shows real-time progress in notification bar.
 */
class NemoForegroundService : Service() {

    private var job: Job? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val task = intent?.getStringExtra(EXTRA_TASK)
        if (task.isNullOrBlank()) {
            stopSelf()
            return START_NOT_STICKY
        }

        startForeground(NOTIFICATION_ID, buildNotification("Starting...", 0))
        executeTask(task)

        return START_STICKY
    }

    private fun executeTask(task: String) {
        job?.cancel()
        job = scope.launch {
            val app = NemoApp.instance
            try {
                val llm = app.getLLM()
                val settings = app.appState.loadSettings()
                val agent = PhoneAgent(
                    llm = llm,
                    maxSteps = settings.maxSteps,
                    actionDelayMs = settings.actionDelayMs,
                    safetyEnabled = settings.safetyEnabled,
                    homeostasisEnabled = settings.homeostasisEnabled,
                    immuneEnabled = settings.immuneEnabled,
                    inertiaEnabled = settings.inertiaEnabled,
                    explorerEnabled = settings.explorerEnabled,
                    circadianEnabled = settings.circadianEnabled,
                )
                agent.initPersistence(applicationContext)

                val result = agent.execute(task) { step ->
                    updateNotification(
                        "Step ${step.step}: ${step.action}",
                        step.step,
                    )
                }

                showCompletionNotification(result)
            } catch (e: Exception) {
                showCompletionNotification(
                    TaskResult(TaskStatus.FAILED, "Error: ${e.message}")
                )
            } finally {
                stopSelf()
            }
        }
    }

    override fun onDestroy() {
        job?.cancel()
        scope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // --- Notifications ---

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(CHANNEL_ID, "Nemo Agent", NotificationManager.IMPORTANCE_LOW)
        channel.description = "Shows agent task progress"
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(text: String, step: Int): Notification {
        val contentIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return builder
            .setContentTitle("Nemo Agent")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentIntent(contentIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String, step: Int) {
        val manager = getSystemService(NotificationManager::class.java) ?: return
        manager.notify(NOTIFICATION_ID, buildNotification(text, step))
    }

    private fun showCompletionNotification(result: TaskResult) {
        val manager = getSystemService(NotificationManager::class.java) ?: return
        val icon = if (result.status == TaskStatus.COMPLETED)
            android.R.drawable.stat_sys_download_done
        else
            android.R.drawable.stat_notify_error

        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        val notification = builder
            .setContentTitle("Task ${result.status.name.lowercase()}")
            .setContentText(result.summary)
            .setSmallIcon(icon)
            .setAutoCancel(true)
            .build()

        manager.notify(NOTIFICATION_ID + 1, notification)
    }

    companion object {
        private const val CHANNEL_ID = "nemo_agent_runtime"
        private const val NOTIFICATION_ID = 1001
        const val EXTRA_TASK = "task"

        fun start(context: Context, task: String) {
            val intent = Intent(context, NemoForegroundService::class.java).apply {
                putExtra(EXTRA_TASK, task)
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                context.startForegroundService(intent)
            } else {
                context.startService(intent)
            }
        }
    }
}
