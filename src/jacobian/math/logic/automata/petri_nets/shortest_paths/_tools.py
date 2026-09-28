"""Petri-net all-shortest-sequences operation declaration."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.logic.automata.petri_nets.shortest_paths._models import (
    ShortestFiringSequencesRequest,
    ShortestFiringSequencesResult,
)
from jacobian.math.logic.automata.petri_nets.shortest_paths.operations import (
    shortest_firing_sequences,
)


def compute_shortest_firing_sequences(
    request: ShortestFiringSequencesRequest,
) -> ShortestFiringSequencesResult:
    return shortest_firing_sequences(request.source_graph, request.target_marking)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="petri_net.reachability.shortest_sequences.compute",
        title="Find all shortest Petri-net firing sequences",
        description=(
            "Given a complete finite Petri-net reachability graph and target marking, "
            "return every minimum-length transition-index sequence. Distinct transition "
            "labels remain distinct even when they connect the same pair of markings. "
            "Incomplete graphs and families beyond the exact output bound are rejected."
        ),
        request_type=ShortestFiringSequencesRequest,
        result_type=ShortestFiringSequencesResult,
        run=compute_shortest_firing_sequences,
        tags=("petri-net", "reachability", "shortest-paths", "exact"),
        discovery_terms=(
            "all shortest Petri-net firing sequences",
            "all shortest transition-labelled paths to a marking",
            "shortest reachability witness family",
        ),
        examples=(
            OperationExample(
                name="parallel_transition_labels",
                description=(
                    "Two distinct transitions reach the same target in one firing, "
                    "so both shortest transition words are returned."
                ),
                input={
                    "source_graph": {
                        "net": {
                            "place_count": 2,
                            "transition_count": 2,
                            "pre": [[1, 1], [0, 0]],
                            "post": [[0, 0], [1, 1]],
                        },
                        "initial_marking": {"tokens": [1, 0]},
                        "max_states": 2,
                        "states": [
                            {
                                "state_index": 0,
                                "place_axis": [0, 1],
                                "marking": {"tokens": [1, 0]},
                            },
                            {
                                "state_index": 1,
                                "place_axis": [0, 1],
                                "marking": {"tokens": [0, 1]},
                            },
                        ],
                        "edges": [
                            {"source_state": 0, "transition": 0, "target_state": 1},
                            {"source_state": 0, "transition": 1, "target_state": 1},
                        ],
                        "truncated": False,
                    },
                    "target_marking": {"tokens": [0, 1]},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
