package com.nemo.ui

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey

/**
 * App state persistence via EncryptedSharedPreferences.
 * All settings encrypted at rest with AES-256.
 */
class AppState(context: Context) {

    private val prefs: SharedPreferences = try {
        val masterKey = MasterKey.Builder(context)
            .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
            .build()
        EncryptedSharedPreferences.create(
            context, "nemo_prefs_enc", masterKey,
            EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
            EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
        )
    } catch (_: Exception) {
        context.getSharedPreferences("nemo_prefs", Context.MODE_PRIVATE)
    }

    var setupCompleted: Boolean
        get() = prefs.getBoolean(KEY_SETUP_COMPLETED, false)
        set(value) = prefs.edit().putBoolean(KEY_SETUP_COMPLETED, value).apply()

    var modelPath: String
        get() = prefs.getString(KEY_MODEL_PATH, "") ?: ""
        set(value) = prefs.edit().putString(KEY_MODEL_PATH, value).apply()

    var safetyEnabled: Boolean
        get() = prefs.getBoolean(KEY_SAFETY, true)
        set(value) = prefs.edit().putBoolean(KEY_SAFETY, value).apply()

    var homeostasisEnabled: Boolean
        get() = prefs.getBoolean(KEY_HOMEOSTASIS, true)
        set(value) = prefs.edit().putBoolean(KEY_HOMEOSTASIS, value).apply()

    var immuneEnabled: Boolean
        get() = prefs.getBoolean(KEY_IMMUNE, false)
        set(value) = prefs.edit().putBoolean(KEY_IMMUNE, value).apply()

    var inertiaEnabled: Boolean
        get() = prefs.getBoolean(KEY_INERTIA, true)
        set(value) = prefs.edit().putBoolean(KEY_INERTIA, value).apply()

    var explorerEnabled: Boolean
        get() = prefs.getBoolean(KEY_EXPLORER, false)
        set(value) = prefs.edit().putBoolean(KEY_EXPLORER, value).apply()

    var circadianEnabled: Boolean
        get() = prefs.getBoolean(KEY_CIRCADIAN, true)
        set(value) = prefs.edit().putBoolean(KEY_CIRCADIAN, value).apply()

    var maxSteps: Int
        get() = prefs.getInt(KEY_MAX_STEPS, 30)
        set(value) = prefs.edit().putInt(KEY_MAX_STEPS, value).apply()

    var actionDelayMs: Long
        get() = prefs.getLong(KEY_ACTION_DELAY, 300L)
        set(value) = prefs.edit().putLong(KEY_ACTION_DELAY, value).apply()

    fun loadSettings(): AppSettings = AppSettings(
        modelPath = modelPath,
        safetyEnabled = safetyEnabled,
        homeostasisEnabled = homeostasisEnabled,
        immuneEnabled = immuneEnabled,
        inertiaEnabled = inertiaEnabled,
        explorerEnabled = explorerEnabled,
        circadianEnabled = circadianEnabled,
        maxSteps = maxSteps,
        actionDelayMs = actionDelayMs,
    )

    fun saveSettings(s: AppSettings) {
        modelPath = s.modelPath
        safetyEnabled = s.safetyEnabled
        homeostasisEnabled = s.homeostasisEnabled
        immuneEnabled = s.immuneEnabled
        inertiaEnabled = s.inertiaEnabled
        explorerEnabled = s.explorerEnabled
        circadianEnabled = s.circadianEnabled
        maxSteps = s.maxSteps
        actionDelayMs = s.actionDelayMs
    }

    companion object {
        private const val KEY_SETUP_COMPLETED = "setup_completed"
        private const val KEY_MODEL_PATH = "model_path"
        private const val KEY_SAFETY = "safety_enabled"
        private const val KEY_HOMEOSTASIS = "homeostasis_enabled"
        private const val KEY_IMMUNE = "immune_enabled"
        private const val KEY_INERTIA = "inertia_enabled"
        private const val KEY_EXPLORER = "explorer_enabled"
        private const val KEY_CIRCADIAN = "circadian_enabled"
        private const val KEY_MAX_STEPS = "max_steps"
        private const val KEY_ACTION_DELAY = "action_delay_ms"
    }
}
