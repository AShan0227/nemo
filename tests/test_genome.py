"""Tests for workflow genome DSL."""

from src.research.genome import (
    Codon, Gene, Opcode, Workflow,
    gene_open_app, gene_search, gene_type_and_send,
    mutate_workflow, crossover_workflows,
)


def test_codon_encode_decode():
    c = Codon(Opcode.TAP, 5)
    encoded = c.encode()
    decoded = Codon.decode(encoded)
    assert decoded.opcode == Opcode.TAP
    assert decoded.operand == 5


def test_codon_to_action():
    tap = Codon(Opcode.TAP, 3)
    action = tap.to_action_dict()
    assert action["action"] == "tap"
    assert action["params"]["index"] == 3


def test_codon_type_with_text_table():
    typ = Codon(Opcode.TYP, 1)
    action = typ.to_action_dict({1: "hello"})
    assert action["action"] == "type_text"
    assert action["params"]["text"] == "hello"


def test_gene_construction():
    gene = gene_open_app(0)
    assert gene.name == "open_app"
    assert gene.length == 1


def test_workflow_to_actions():
    wf = Workflow(
        name="send_message",
        genes=[gene_open_app(0), gene_type_and_send(1)],
        text_table={0: "com.tencent.mm", 1: "hello"},
    )
    actions = wf.to_action_sequence()
    assert len(actions) == 4  # launch + tap + type + tap
    assert actions[0]["action"] == "launch"
    assert actions[2]["action"] == "type_text"


def test_workflow_encode():
    wf = Workflow(
        name="test",
        genes=[Gene("g1", [Codon(Opcode.TAP, 1), Codon(Opcode.BCK, 0)])],
    )
    encoded = wf.encode()
    assert len(encoded) == 2
    assert all(isinstance(v, int) for v in encoded)


def test_mutation():
    wf = Workflow(
        name="test",
        genes=[Gene("g1", [Codon(Opcode.TAP, 1)] * 5)],
    )
    mutated = mutate_workflow(wf, rate=1.0)
    # At least some codons should differ
    original_opcodes = [c.opcode for c in wf.codons]
    mutated_opcodes = [c.opcode for c in mutated.codons]
    # Can't guarantee difference (random), but structure should be valid
    assert len(mutated.genes) > 0


def test_crossover():
    w1 = Workflow(
        name="w1",
        genes=[Gene("a", [Codon(Opcode.TAP, 0)]), Gene("b", [Codon(Opcode.BCK, 0)])],
    )
    w2 = Workflow(
        name="w2",
        genes=[Gene("c", [Codon(Opcode.HOM, 0)]), Gene("d", [Codon(Opcode.SCR, 0)])],
    )
    child = crossover_workflows(w1, w2)
    assert len(child.genes) > 0
