"""Typed declarations for the hypergraph colouring decision operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.coloring.hypergraph_coloring._models import (
    HypergraphColoringRequest,
    HypergraphColoringResult,
)
from jacobian.math.graphs.coloring.hypergraph_coloring.operations import (
    decide_hypergraph_coloring,
)


def _decide(request: HypergraphColoringRequest) -> HypergraphColoringResult:
    return decide_hypergraph_coloring(request.hypergraph, request.palette_size)


TOOLS: MathTools = (
    MathTool(
        operation_id="hypergraph.coloring.non_monochromatic.decide",
        title="Decide bounded non-monochromatic vertex-colourability of a finite hypergraph",
        description=(
            "Given a finite hypergraph and a positive palette size q, decide "
            "whether H has a vertex q-colouring in which no hyperedge is "
            "monochromatic. If one exists, return a canonical colouring."
        ),
        request_type=HypergraphColoringRequest,
        result_type=HypergraphColoringResult,
        run=_decide,
        tags=("hypergraph", "coloring", "exact"),
        examples=(
            example(
                "simple",
                "A 3-vertex 2-edge hypergraph with 2 colours.",
                {
                    "hypergraph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [
                            ["e0", ["a", "b"]],
                            ["e1", ["b", "c"]],
                        ],
                    },
                    "palette_size": 2,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
