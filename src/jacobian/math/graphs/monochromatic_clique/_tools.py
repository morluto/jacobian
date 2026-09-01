"""Monochromatic clique hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.graphs.monochromatic_clique._models import (
    MonochromaticCliqueHypergraphRequest,
    MonochromaticCliqueHypergraphResult,
)
from jacobian.math.graphs.monochromatic_clique.operations import (
    construct_monochromatic_clique_hypergraph,
)


def compute_monochromatic_clique_hypergraph(
    request: MonochromaticCliqueHypergraphRequest,
) -> MonochromaticCliqueHypergraphResult:
    return construct_monochromatic_clique_hypergraph(
        request.colored_graph, request.clique_order
    )


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.edge_colored.monochromatic_clique_hypergraph.construct",
        title="Construct the monochromatic-clique hypergraph of an edge-coloured graph",
        description=(
            "Given a bounded complete edge-coloured simple graph and an integer "
            "t >= 2, return the canonical t-uniform FiniteHypergraph whose "
            "hyperedges are exactly the t-element vertex sets inducing a "
            "monochromatic clique in the source."
        ),
        request_type=MonochromaticCliqueHypergraphRequest,
        result_type=MonochromaticCliqueHypergraphResult,
        run=compute_monochromatic_clique_hypergraph,
        tags=("graph", "ramsey", "exact"),
        examples=(
            OperationExample(
                name="all_red_k4_t3",
                description="All-red K4 with target clique order 3.",
                input={
                    "colored_graph": {
                        "graph": {
                            "vertices": ["0", "1", "2", "3"],
                            "edges": [
                                ["0", "1"],
                                ["0", "2"],
                                ["0", "3"],
                                ["1", "2"],
                                ["1", "3"],
                                ["2", "3"],
                            ],
                        },
                        "edge_colors": ["red", "red", "red", "red", "red", "red"],
                    },
                    "clique_order": 3,
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
