package com.nemo.knowledge

import androidx.room.*
import java.util.PriorityQueue
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

// ---------------------------------------------------------------------------
// Room Entities
// ---------------------------------------------------------------------------

@Entity(tableName = "screen_nodes")
data class ScreenNode(
    @PrimaryKey val stateId: String,
    val activity: String = "",
    @ColumnInfo(name = "package_name") val packageName: String = "",
    val description: String = "",
    val visitCount: Int = 0,
    val uiHash: String = ""
)

@Entity(
    tableName = "action_edges",
    foreignKeys = [
        ForeignKey(entity = ScreenNode::class, parentColumns = ["stateId"], childColumns = ["fromState"]),
        ForeignKey(entity = ScreenNode::class, parentColumns = ["stateId"], childColumns = ["toState"])
    ],
    indices = [Index("fromState"), Index("toState")]
)
data class ActionEdge(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val fromState: String,
    val toState: String,
    val actionType: String,
    val actionParams: String = "", // JSON string
    var cost: Double = 1.0,
    var successCount: Int = 0,
    var failureCount: Int = 0,
    var pheromone: Double = 1.0
) {
    val successRate: Double
        get() {
            val total = successCount + failureCount
            return if (total > 0) successCount.toDouble() / total else 0.5
        }
}

// ---------------------------------------------------------------------------
// Room DAO
// ---------------------------------------------------------------------------

@Dao
interface ScreenGraphDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertNode(node: ScreenNode)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertEdge(edge: ActionEdge): Long

    @Update
    suspend fun updateEdge(edge: ActionEdge)

    @Query("SELECT * FROM screen_nodes WHERE stateId = :stateId")
    suspend fun getNode(stateId: String): ScreenNode?

    @Query("SELECT * FROM screen_nodes")
    suspend fun getAllNodes(): List<ScreenNode>

    @Query("SELECT * FROM action_edges WHERE fromState = :stateId")
    suspend fun getEdgesFrom(stateId: String): List<ActionEdge>

    @Query("SELECT * FROM action_edges WHERE fromState = :from AND toState = :to AND actionType = :type LIMIT 1")
    suspend fun findEdge(from: String, to: String, type: String): ActionEdge?

    @Query("SELECT * FROM action_edges")
    suspend fun getAllEdges(): List<ActionEdge>

    @Query("SELECT COUNT(*) FROM screen_nodes")
    suspend fun nodeCount(): Int

    @Query("SELECT COUNT(*) FROM action_edges")
    suspend fun edgeCount(): Int
}

// ---------------------------------------------------------------------------
// Room Database
// ---------------------------------------------------------------------------

@Database(entities = [ScreenNode::class, ActionEdge::class], version = 1, exportSchema = false)
abstract class ScreenGraphDatabase : RoomDatabase() {
    abstract fun graphDao(): ScreenGraphDao
}

// ---------------------------------------------------------------------------
// In-Memory ScreenGraph (A* + ACO, mirroring Python graph.py)
// ---------------------------------------------------------------------------

class ScreenGraph {
    private val nodes = mutableMapOf<String, ScreenNode>()
    private val edges = mutableMapOf<String, MutableList<ActionEdge>>()

    fun addNode(node: ScreenNode) {
        nodes[node.stateId] = node
    }

    fun addEdge(edge: ActionEdge) {
        edges.getOrPut(edge.fromState) { mutableListOf() }.add(edge)
    }

    fun getNeighbors(stateId: String): List<ActionEdge> = edges[stateId] ?: emptyList()

    fun getNode(stateId: String): ScreenNode? = nodes[stateId]

    val nodeCount: Int get() = nodes.size
    val edgeCount: Int get() = edges.values.sumOf { it.size }

    /**
     * Record a transition outcome, building the graph incrementally.
     * Updates pheromone and cost based on success/failure.
     */
    fun recordTransition(fromState: String, toState: String, actionType: String, success: Boolean) {
        if (fromState !in nodes) addNode(ScreenNode(stateId = fromState))
        if (toState !in nodes) addNode(ScreenNode(stateId = toState))

        val neighbors = getNeighbors(fromState)
        val existing = neighbors.find { it.toState == toState && it.actionType == actionType }

        val edge = existing ?: ActionEdge(
            fromState = fromState,
            toState = toState,
            actionType = actionType
        ).also { addEdge(it) }

        if (success) {
            edge.successCount++
            edge.pheromone = min(edge.pheromone * 1.1, 10.0)
        } else {
            edge.failureCount++
            edge.pheromone = max(edge.pheromone * 0.9, 0.1)
        }
        edge.cost = 1.0 / max(edge.successRate, 0.01)
    }

    /**
     * A* search from start to goal screen state.
     * Returns optimal path as list of edges, or null if unreachable.
     */
    fun aStar(
        start: String,
        goal: String,
        heuristic: (String, String) -> Double = { _, _ -> 0.0 }
    ): List<ActionEdge>? {
        if (start == goal) return emptyList()

        data class Entry(val cost: Double, val state: String, val path: List<ActionEdge>)

        val queue = PriorityQueue<Entry>(compareBy { it.cost })
        queue.add(Entry(0.0, start, emptyList()))
        val visited = mutableSetOf<String>()

        while (queue.isNotEmpty()) {
            val (cost, current, path) = queue.poll()
            if (current == goal) return path
            if (current in visited) continue
            visited.add(current)

            for (edge in getNeighbors(current)) {
                if (edge.toState !in visited) {
                    val newCost = cost + edge.cost
                    val h = heuristic(edge.toState, goal)
                    queue.add(Entry(newCost + h, edge.toState, path + edge))
                }
            }
        }
        return null // no path found
    }

    /**
     * ACO action selection with pseudo-random proportional rule.
     * With probability q0, greedily pick best (exploitation).
     * With probability 1-q0, sample proportionally (exploration).
     */
    fun selectActionAco(
        stateId: String,
        alpha: Double = 1.0,
        beta: Double = 2.0,
        q0: Double = 0.9
    ): ActionEdge? {
        val neighbors = getNeighbors(stateId)
        if (neighbors.isEmpty()) return null
        if (neighbors.size == 1) return neighbors[0]

        val scores = neighbors.map { edge ->
            val tau = Math.pow(edge.pheromone, alpha)
            val eta = Math.pow(1.0 / max(edge.cost, 0.01), beta)
            tau * eta
        }

        return if (Random.nextDouble() < q0) {
            // Exploitation: pick best
            neighbors[scores.indices.maxByOrNull { scores[it] } ?: 0]
        } else {
            // Exploration: roulette wheel
            val total = scores.sum()
            if (total == 0.0) return neighbors.random()
            val pick = Random.nextDouble() * total
            var cumulative = 0.0
            for (i in scores.indices) {
                cumulative += scores[i]
                if (cumulative >= pick) return neighbors[i]
            }
            neighbors.last()
        }
    }

    /** ACO pheromone evaporation — prevents lock-in to suboptimal paths. */
    fun evaporatePheromones(rate: Double = 0.05) {
        for (edgeList in edges.values) {
            for (edge in edgeList) {
                edge.pheromone = max(edge.pheromone * (1.0 - rate), 0.1)
            }
        }
    }

    /** Sync in-memory graph to Room database. */
    suspend fun saveTo(dao: ScreenGraphDao) {
        for (node in nodes.values) dao.insertNode(node)
        for (edgeList in edges.values) {
            for (edge in edgeList) dao.insertEdge(edge)
        }
    }

    /** Load graph from Room database. */
    suspend fun loadFrom(dao: ScreenGraphDao) {
        nodes.clear()
        edges.clear()
        for (node in dao.getAllNodes()) nodes[node.stateId] = node
        for (edge in dao.getAllEdges()) {
            edges.getOrPut(edge.fromState) { mutableListOf() }.add(edge)
        }
    }
}
