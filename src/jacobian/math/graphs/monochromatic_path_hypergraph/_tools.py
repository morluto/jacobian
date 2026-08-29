"""Typed declarations for the monochromatic path hypergraph operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.monochromatic_path_hypergraph._models import (
    MonochromaticPathRequest,
    MonochromaticPathHypergraphResult,
)
from jacobian.math.graphs.monochromatic_path_hypergraph.operations import (
    construct_monochromatic_path_hypergraphs,
)


def _construct(request: MonochromaticPathRequest) -> MonochromaticPathHypergraphResult:
    return construct_monochromatic_path_hypergraphs(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_colored.monochromatic_path_hypergraphs.construct",
        title="Construct monochromatic simple-path candidate hypergraphs",
        description=(
            "Given a bounded edge-coloured simple graph, return one canonical "
            "FiniteHypergraph per colour whose edges are the vertex supports of "
            "simple paths using only that colour."
        ),
        request_type=MonochromaticPathRequest,
        result_type=MonochromaticPathHypergraphResult,
        run=_construct,
        tags=("graph", "monochromatic", "path", "hypergraph", "exact"),
        examples=(
            example(
                "two_color_path",
                "Two-colour path graph.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [
                            {"left": "a", "right": "b", "color": "red"},
                            {"left": "b", "right": "c", "color": "blue"},
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
