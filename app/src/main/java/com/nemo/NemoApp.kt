package com.nemo

import android.app.Application
import com.nemo.knowledge.GraphPersistence
import com.nemo.model.ModelManager
import com.nemo.model.OnDeviceLLM
import com.nemo.service.NemoAccessibilityService
import com.nemo.ui.AppState
import kotlinx.coroutines.*

/**
 * Application singleton — global initialization + persistent LLM instance.
 * Model loads once and stays in memory across tasks.
 */
class NemoApp : Application() {

    lateinit var appState: AppState; private set
    lateinit var modelManager: ModelManager; private set
    lateinit var graphPersistence: GraphPersistence; private set

    private var _llm: OnDeviceLLM? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    val isAccessibilityEnabled: Boolean
        get() = NemoAccessibilityService.instance != null

    val isModelReady: Boolean
        get() = _llm?.isLoaded == true

    val isReady: Boolean
        get() = isAccessibilityEnabled && isModelReady

    override fun onCreate() {
        super.onCreate()
        instance = this
        appState = AppState(this)
        modelManager = ModelManager(this)
        graphPersistence = GraphPersistence(this)
    }

    /**
     * Get or create the singleton LLM instance.
     * First call triggers model load (~200-300ms).
     */
    suspend fun getLLM(): OnDeviceLLM {
        _llm?.let { if (it.isLoaded) return it }
        val path = appState.modelPath.ifBlank { modelManager.defaultModelPath() }
        val llm = OnDeviceLLM(this, path)
        llm.load()
        _llm = llm
        return llm
    }

    fun releaseLLM() {
        _llm?.close()
        _llm = null
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_RUNNING_CRITICAL) {
            releaseLLM()
        }
    }

    companion object {
        lateinit var instance: NemoApp; private set
    }
}
