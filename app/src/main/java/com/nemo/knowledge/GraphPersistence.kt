package com.nemo.knowledge

import android.content.Context
import androidx.room.Room
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Manages ScreenGraph lifecycle with Room persistence.
 * Load graph from SQLite on startup, save after each task.
 */
class GraphPersistence(context: Context) {

    private val db = Room.databaseBuilder(
        context.applicationContext,
        ScreenGraphDatabase::class.java,
        "nemo_screen_graph"
    ).fallbackToDestructiveMigration().build()

    val dao: ScreenGraphDao = db.graphDao()

    /**
     * Load entire graph from Room into an in-memory ScreenGraph.
     */
    suspend fun loadGraph(): ScreenGraph = withContext(Dispatchers.IO) {
        val graph = ScreenGraph()
        graph.loadFrom(dao)
        graph
    }

    /**
     * Save in-memory ScreenGraph to Room.
     */
    suspend fun saveGraph(graph: ScreenGraph) = withContext(Dispatchers.IO) {
        graph.saveTo(dao)
    }

    /**
     * Import a recorded action sequence into the graph.
     * Each consecutive pair of (screenHash, action) creates a graph edge.
     */
    suspend fun importRecording(
        steps: List<RecordingStep>,
    ): Int = withContext(Dispatchers.IO) {
        var edgesAdded = 0
        for (i in 0 until steps.size - 1) {
            val from = steps[i]
            val to = steps[i + 1]
            val existing = dao.findEdge(from.screenHash, to.screenHash, from.actionType)
            if (existing != null) {
                existing.successCount++
                existing.pheromone = minOf(existing.pheromone * 1.1, 10.0)
                existing.cost = 1.0 / maxOf(existing.successRate, 0.01)
                dao.updateEdge(existing)
            } else {
                dao.insertNode(ScreenNode(stateId = from.screenHash, activity = from.activity))
                dao.insertNode(ScreenNode(stateId = to.screenHash, activity = to.activity))
                dao.insertEdge(ActionEdge(
                    fromState = from.screenHash,
                    toState = to.screenHash,
                    actionType = from.actionType,
                    successCount = 1,
                    pheromone = 1.1,
                ))
                edgesAdded++
            }
        }
        edgesAdded
    }
}

data class RecordingStep(
    val screenHash: String,
    val activity: String = "",
    val actionType: String,
)
