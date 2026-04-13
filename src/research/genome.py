"""Workflow Genome DSL — compact encoding of task execution workflows.

Encodes workflows as "genomes" that can be crossed over, mutated, and evolved.
Inspired by DNA's hierarchical encoding: codons -> genes -> chromosomes -> genome.

Hierarchy:
- Codon (atomic operation): TAP, SWP, TYP, WAT, VER, BCK, HOM, LCH
- Gene (reusable sub-task): open_app, search, scroll_find, login
- Chromosome (complete workflow): e.g., send WeChat message
- Genome (multi-workflow): complex multi-app tasks

Each codon encodes as a compact instruction with opcode + operands.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from loguru import logger


class Opcode(IntEnum):
    """Atomic operations (codons)."""
    NOP = 0   # no operation
    TAP = 1   # tap(index)
    SWP = 2   # swipe(direction)
    TYP = 3   # type(text_ref)
    WAT = 4   # wait(ms)
    VER = 5   # verify(condition_ref)
    BCK = 6   # back
    HOM = 7   # home
    LCH = 8   # launch(package_ref)
    SCR = 9   # scroll(direction)
    DON = 10  # done


@dataclass
class Codon:
    """Atomic instruction in the workflow genome."""
    opcode: Opcode
    operand: int = 0  # index, direction, or reference ID
    label: str = ""   # human-readable description

    def encode(self) -> int:
        """Encode as 16-bit integer: [4-bit opcode][12-bit operand]."""
        return (self.opcode.value << 12) | (self.operand & 0xFFF)

    @staticmethod
    def decode(value: int) -> Codon:
        """Decode from 16-bit integer."""
        opcode = Opcode((value >> 12) & 0xF)
        operand = value & 0xFFF
        return Codon(opcode=opcode, operand=operand)

    def to_action_dict(self, text_table: Dict[int, str] = None) -> Dict:
        """Convert codon to agent action format."""
        text_table = text_table or {}
        if self.opcode == Opcode.TAP:
            return {"action": "tap", "params": {"index": self.operand}}
        elif self.opcode == Opcode.TYP:
            text = text_table.get(self.operand, "")
            return {"action": "type_text", "params": {"index": 0, "text": text}}
        elif self.opcode == Opcode.SWP or self.opcode == Opcode.SCR:
            direction = "down" if self.operand == 0 else "up"
            return {"action": "scroll", "params": {"direction": direction}}
        elif self.opcode == Opcode.BCK:
            return {"action": "back", "params": {}}
        elif self.opcode == Opcode.HOM:
            return {"action": "home", "params": {}}
        elif self.opcode == Opcode.LCH:
            pkg = text_table.get(self.operand, "")
            return {"action": "launch", "params": {"package": pkg}}
        elif self.opcode == Opcode.WAT:
            return {"action": "wait", "params": {"ms": self.operand * 100}}
        elif self.opcode == Opcode.DON:
            return {"action": "done", "params": {"summary": "workflow complete"}}
        return {"action": "wait", "params": {"ms": 100}}


@dataclass
class Gene:
    """Reusable sub-task (sequence of codons)."""
    name: str
    codons: List[Codon]
    description: str = ""

    def encode(self) -> List[int]:
        return [c.encode() for c in self.codons]

    @property
    def length(self) -> int:
        return len(self.codons)


@dataclass
class Workflow:
    """Complete task workflow (sequence of genes)."""
    name: str
    genes: List[Gene]
    text_table: Dict[int, str] = field(default_factory=dict)
    fitness: float = 0.0

    @property
    def codons(self) -> List[Codon]:
        """Flatten all genes into codon sequence."""
        result = []
        for gene in self.genes:
            result.extend(gene.codons)
        return result

    def encode(self) -> List[int]:
        return [c.encode() for c in self.codons]

    def to_action_sequence(self) -> List[Dict]:
        """Convert to a list of agent action dicts."""
        return [c.to_action_dict(self.text_table) for c in self.codons]

    @property
    def total_length(self) -> int:
        return sum(g.length for g in self.genes)


# --- Genetic operators ---

def mutate_workflow(
    workflow: Workflow, rate: float = 0.1
) -> Workflow:
    """Mutate a workflow by randomly changing codons."""
    from copy import deepcopy
    mutated = deepcopy(workflow)

    for gene in mutated.genes:
        for i, codon in enumerate(gene.codons):
            if random.random() < rate:
                mutation_type = random.choice(["operand", "opcode", "insert", "delete"])

                if mutation_type == "operand":
                    gene.codons[i] = Codon(codon.opcode, random.randint(0, 15))
                elif mutation_type == "opcode":
                    new_op = random.choice(list(Opcode))
                    gene.codons[i] = Codon(new_op, codon.operand)
                elif mutation_type == "insert" and len(gene.codons) < 20:
                    new_codon = Codon(random.choice(list(Opcode)), random.randint(0, 10))
                    gene.codons.insert(i, new_codon)
                elif mutation_type == "delete" and len(gene.codons) > 1:
                    gene.codons.pop(i)
                    break  # avoid index issues

    return mutated


def crossover_workflows(
    w1: Workflow, w2: Workflow
) -> Workflow:
    """Single-point crossover at gene level."""
    from copy import deepcopy

    if len(w1.genes) <= 1 or len(w2.genes) <= 1:
        return deepcopy(w1)

    point1 = random.randint(1, len(w1.genes) - 1)
    point2 = random.randint(1, len(w2.genes) - 1)

    child_genes = deepcopy(w1.genes[:point1]) + deepcopy(w2.genes[point2:])
    text_table = {**w1.text_table, **w2.text_table}

    return Workflow(
        name=f"{w1.name}x{w2.name}",
        genes=child_genes,
        text_table=text_table,
    )


# --- Predefined gene library ---

def gene_open_app(package_ref: int) -> Gene:
    return Gene("open_app", [Codon(Opcode.LCH, package_ref, f"launch app #{package_ref}")])


def gene_search(text_ref: int) -> Gene:
    return Gene("search", [
        Codon(Opcode.TAP, 0, "tap search bar"),
        Codon(Opcode.TYP, text_ref, f"type search text #{text_ref}"),
        Codon(Opcode.TAP, 0, "tap search button"),
    ])


def gene_scroll_find(target_index: int, max_scrolls: int = 3) -> Gene:
    codons = []
    for _ in range(max_scrolls):
        codons.append(Codon(Opcode.SCR, 0, "scroll down"))
        codons.append(Codon(Opcode.WAT, 5, "wait 500ms"))
    codons.append(Codon(Opcode.TAP, target_index, f"tap found item #{target_index}"))
    return Gene("scroll_find", codons)


def gene_type_and_send(text_ref: int, input_index: int = 0, send_index: int = 1) -> Gene:
    return Gene("type_and_send", [
        Codon(Opcode.TAP, input_index, "tap input field"),
        Codon(Opcode.TYP, text_ref, f"type message #{text_ref}"),
        Codon(Opcode.TAP, send_index, "tap send"),
    ])
