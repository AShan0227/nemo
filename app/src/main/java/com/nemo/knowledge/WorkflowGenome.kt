package com.nemo.knowledge

import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/**
 * Compact DNA-like encoding for mobile task workflows.
 *
 * Each action is encoded as a 16-bit Codon: [4-bit opcode | 12-bit operand].
 * Codons form Genes (reusable sub-tasks), Genes form Workflows (complete tasks).
 * Supports mutation, crossover, and evolution via genetic algorithms.
 *
 * Ported from Python: src/research/genome.py
 */

// ---------------------------------------------------------------------------
// Opcode definitions
// ---------------------------------------------------------------------------

enum class Opcode(val code: Int) {
    NOP(0), TAP(1), SWP(2), TYP(3), WAT(4), VER(5),
    BCK(6), HOM(7), LCH(8), SCR(9), DON(10);

    companion object {
        fun fromCode(code: Int): Opcode = entries.firstOrNull { it.code == code } ?: NOP
    }
}

// ---------------------------------------------------------------------------
// Codon — atomic instruction
// ---------------------------------------------------------------------------

data class Codon(
    val opcode: Opcode,
    val operand: Int = 0, // 0-4095
    val label: String = ""
) {
    /** Encode as 16-bit integer: [opcode(4) | operand(12)] */
    fun encode(): Int = (opcode.code shl 12) or (operand and 0xFFF)

    /** Convert to agent action dict format. */
    fun toActionMap(textTable: Map<Int, String> = emptyMap()): Map<String, Any> = when (opcode) {
        Opcode.TAP -> mapOf("action" to "tap", "params" to mapOf("index" to operand))
        Opcode.TYP -> mapOf(
            "action" to "type_text",
            "params" to mapOf("index" to 0, "text" to (textTable[operand] ?: ""))
        )
        Opcode.SWP -> mapOf("action" to "scroll", "params" to mapOf("direction" to "down"))
        Opcode.SCR -> mapOf("action" to "scroll", "params" to mapOf("direction" to "up"))
        Opcode.BCK -> mapOf("action" to "back", "params" to emptyMap<String, Any>())
        Opcode.HOM -> mapOf("action" to "home", "params" to emptyMap<String, Any>())
        Opcode.LCH -> mapOf(
            "action" to "launch",
            "params" to mapOf("package" to (textTable[operand] ?: ""))
        )
        Opcode.WAT -> mapOf("action" to "wait", "params" to mapOf("ms" to operand * 100))
        Opcode.DON -> mapOf("action" to "done", "params" to mapOf("summary" to "workflow complete"))
        Opcode.VER -> mapOf("action" to "wait", "params" to mapOf("ms" to 500))
        Opcode.NOP -> mapOf("action" to "wait", "params" to mapOf("ms" to 0))
    }

    companion object {
        fun decode(encoded: Int): Codon {
            val opcode = Opcode.fromCode((encoded shr 12) and 0xF)
            val operand = encoded and 0xFFF
            return Codon(opcode, operand)
        }
    }
}

// ---------------------------------------------------------------------------
// Gene — reusable sub-task
// ---------------------------------------------------------------------------

data class Gene(
    val name: String,
    val codons: List<Codon>,
    val description: String = ""
) {
    fun encode(): List<Int> = codons.map { it.encode() }
    val length: Int get() = codons.size
}

// ---------------------------------------------------------------------------
// Workflow — complete task genome
// ---------------------------------------------------------------------------

data class Workflow(
    val name: String,
    val genes: List<Gene>,
    val textTable: Map<Int, String> = emptyMap(),
    var fitness: Double = 0.0
) {
    val codons: List<Codon> get() = genes.flatMap { it.codons }
    val totalLength: Int get() = genes.sumOf { it.length }

    fun encode(): List<Int> = codons.map { it.encode() }

    fun toActionSequence(): List<Map<String, Any>> =
        codons.map { it.toActionMap(textTable) }
}

// ---------------------------------------------------------------------------
// Genetic operators
// ---------------------------------------------------------------------------

/**
 * Mutate a workflow with given probability per codon.
 */
fun mutateWorkflow(workflow: Workflow, rate: Double = 0.1): Workflow {
    val newGenes = workflow.genes.map { gene ->
        val newCodons = gene.codons.toMutableList()
        val mutated = mutableListOf<Codon>()

        for (codon in newCodons) {
            if (Random.nextDouble() >= rate) {
                mutated.add(codon)
                continue
            }
            when (Random.nextInt(4)) {
                0 -> mutated.add(codon.copy(operand = Random.nextInt(16))) // tweak operand
                1 -> mutated.add(codon.copy(opcode = Opcode.entries.random())) // change opcode
                2 -> { // insert
                    mutated.add(codon)
                    if (mutated.size < 20) {
                        mutated.add(Codon(Opcode.entries.random(), Random.nextInt(16)))
                    }
                }
                3 -> { /* delete: skip this codon */ }
            }
        }
        if (mutated.isEmpty()) mutated.add(Codon(Opcode.NOP))
        gene.copy(codons = mutated)
    }
    return workflow.copy(genes = newGenes)
}

/**
 * Single-point gene-level crossover between two workflows.
 */
fun crossoverWorkflows(w1: Workflow, w2: Workflow): Workflow {
    if (w1.genes.isEmpty() || w2.genes.isEmpty()) return w1

    val cut1 = Random.nextInt(w1.genes.size)
    val cut2 = Random.nextInt(w2.genes.size)

    val childGenes = w1.genes.subList(0, cut1 + 1) + w2.genes.subList(cut2, w2.genes.size)
    val mergedTextTable = w1.textTable + w2.textTable // w2 overrides on collision

    return Workflow(
        name = "${w1.name}×${w2.name}",
        genes = childGenes,
        textTable = mergedTextTable
    )
}

// ---------------------------------------------------------------------------
// Gene library — pre-built building blocks
// ---------------------------------------------------------------------------

object GeneLibrary {
    fun openApp(packageRef: Int): Gene = Gene(
        name = "open_app",
        codons = listOf(Codon(Opcode.LCH, packageRef)),
        description = "Launch an app by package reference"
    )

    fun search(textRef: Int): Gene = Gene(
        name = "search",
        codons = listOf(
            Codon(Opcode.TAP, 0, "tap search bar"),
            Codon(Opcode.TYP, textRef, "type query"),
            Codon(Opcode.TAP, 1, "tap search button")
        ),
        description = "Search for text"
    )

    fun scrollFind(targetIndex: Int, maxScrolls: Int = 3): Gene {
        val codons = mutableListOf<Codon>()
        repeat(maxScrolls) {
            codons.add(Codon(Opcode.SCR, 1))
            codons.add(Codon(Opcode.WAT, 5)) // 500ms
        }
        codons.add(Codon(Opcode.TAP, targetIndex))
        return Gene(name = "scroll_find", codons = codons, description = "Scroll to find and tap element")
    }

    fun typeAndSend(textRef: Int, inputIndex: Int = 0, sendIndex: Int = 1): Gene = Gene(
        name = "type_and_send",
        codons = listOf(
            Codon(Opcode.TAP, inputIndex, "tap input"),
            Codon(Opcode.TYP, textRef, "type text"),
            Codon(Opcode.TAP, sendIndex, "tap send")
        ),
        description = "Type text and send"
    )
}
