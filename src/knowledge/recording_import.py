"""Import recorded action traces into the Knowledge Graph.

Converts ActionRecorder recordings into ScreenGraph nodes and edges,
supporting:
1. Auto-generation of graph nodes/edges from a single recording
2. Merging multiple recordings to build a more complete graph
3. Screen state hash association for graph correlation
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.device.actions import ActionRecorder
from src.knowledge.graph import ScreenGraph, ScreenNode


def recording_to_graph(
    recorder: ActionRecorder,
    *,
    screen_hashes: list[str] | None = None,
    graph: ScreenGraph | None = None,
) -> ScreenGraph:
    """Convert an action recording into Knowledge Graph nodes and edges.

    Args:
        recorder: ActionRecorder with recorded action trace.
        screen_hashes: Optional list of screen state hashes, one per screen state.
            Length must be len(recorder.records) + 1 (initial + after each action).
            If not provided, synthetic hashes are generated from timestamps.
        graph: Existing graph to merge into. Creates new graph if None.

    Returns:
        ScreenGraph with nodes and edges from the recording.
    """
    if graph is None:
        graph = ScreenGraph()

    records = recorder.records
    if not records:
        return graph

    # Generate or validate screen hashes
    if screen_hashes is not None:
        if len(screen_hashes) != len(records) + 1:
            logger.warning(
                f"screen_hashes length mismatch: got {len(screen_hashes)}, "
                f"expected {len(records) + 1}. Using synthetic hashes."
            )
            screen_hashes = None

    if screen_hashes is None:
        screen_hashes = [f"screen_{i}" for i in range(len(records) + 1)]

    # Build nodes and edges
    for i, record in enumerate(records):
        from_hash = screen_hashes[i]
        to_hash = screen_hashes[i + 1]

        # Ensure nodes exist
        if from_hash not in graph._nodes:
            graph.add_node(ScreenNode(state_id=from_hash, ui_hash=from_hash))
        if to_hash not in graph._nodes:
            graph.add_node(ScreenNode(state_id=to_hash, ui_hash=to_hash))

        # Record transition
        graph.record_transition(
            from_state=from_hash,
            to_state=to_hash,
            action_type=record.action.type.value,
            success=record.success,
        )

    logger.info(
        f"Imported recording: {len(records)} actions -> "
        f"{len(graph._nodes)} nodes, "
        f"{sum(len(e) for e in graph._edges.values())} edges"
    )
    return graph


def merge_recordings(
    recordings: list[ActionRecorder],
    *,
    screen_hashes_list: list[list[str]] | None = None,
) -> ScreenGraph:
    """Merge multiple recordings into a single Knowledge Graph.

    Overlapping screen states (same hash) are automatically merged,
    creating a more complete navigation graph.
    """
    graph = ScreenGraph()

    for i, recorder in enumerate(recordings):
        hashes = screen_hashes_list[i] if screen_hashes_list and i < len(screen_hashes_list) else None
        recording_to_graph(recorder, screen_hashes=hashes, graph=graph)

    logger.info(
        f"Merged {len(recordings)} recordings -> "
        f"{len(graph._nodes)} nodes, "
        f"{sum(len(e) for e in graph._edges.values())} edges"
    )
    return graph


def import_recording_file(
    path: str | Path,
    *,
    graph: ScreenGraph | None = None,
) -> ScreenGraph:
    """Load a recording file and import into a graph."""
    recorder = ActionRecorder.load(path)
    return recording_to_graph(recorder, graph=graph)
