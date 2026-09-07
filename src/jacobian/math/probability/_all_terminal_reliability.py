"""Wire contract for exact all-terminal graph reliability."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from jacobian._exact import (
    CanonicalRational,
)
from jacobian._models import StrictModel
from jacobian.catalog.models import (
    MathTool,
    OperationExample,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph
from jacobian.math.probability.all_terminal_reliability import (
    AllTerminalReliabilityResult,
    all_terminal_reliability,
)


class AllTerminalReliabilityRequest(StrictModel):
    """One nonempty bounded graph and one uniform exact edge-up probability."""

    model_config = ConfigDict(
        json_schema_extra={
            "description": (
                "Compute all-terminal reliability for one nonempty canonical "
                "simple undirected graph. Every edge is independently open with "
                "the same exact rational `open_probability`. The graph may have "
                "at most 20 edges, bounding each complete enumeration to 2^20 "
                "states. The exact coefficient profile is bounded by the same "
                "finite edge-subset family."
            )
        }
    )

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Nonempty canonical simple undirected graph with at most 20 edges. "
            "Isolated declared vertices participate in the all-terminal event. "
            "The complete coefficient profile has at most 21 entries."
        )
    )
    open_probability: CanonicalRational = Field(
        description=(
            "Uniform independent probability that each graph edge is open, as "
            "an exact canonical rational in [0, 1] with at most 128 digits per "
            "component."
        )
    )
    event: Literal["ALL_VERTICES_CONNECTED"] = "ALL_VERTICES_CONNECTED"


def compute_all_terminal_reliability(
    request: AllTerminalReliabilityRequest,
) -> AllTerminalReliabilityResult:
    return all_terminal_reliability(
        request.graph, request.open_probability.as_fraction()
    )


ALL_TERMINAL_RELIABILITY_OPERATION = MathTool(
    operation_id="probability.graph_reliability.all_terminal.compute",
    title="Exact bounded all-terminal graph reliability",
    description=(
        "Compute the exact probability that the spanning subgraph on every "
        "declared vertex is connected when each edge is independently open with "
        "one uniform rational probability. Return the complete connected-spanning-"
        "subgraph count vector as a source-bound reconstruction value."
    ),
    request_type=AllTerminalReliabilityRequest,
    result_type=AllTerminalReliabilityResult,
    run=compute_all_terminal_reliability,
    tags=(
        "probability",
        "graph",
        "reliability",
        "all-terminal",
        "connected-spanning-subgraph",
        "exact",
        "bounded",
    ),
    examples=(
        OperationExample(
            name="fair_edge_triangle_all_terminal_reliability",
            description=(
                "Compute the exact probability that a fair-edge triangle is "
                "connected on all declared vertices; the graph must be nonempty, "
                "have at most 20 edges, use one rational edge probability, and "
                "fit the retained-result output limit."
            ),
            input={
                "graph": {
                    "vertices": ["a", "b", "c"],
                    "edges": [["a", "b"], ["a", "c"], ["b", "c"]],
                },
                "open_probability": {"num": "1", "den": "2"},
            },
        ),
    ),
)


__all__ = ["ALL_TERMINAL_RELIABILITY_OPERATION"]
