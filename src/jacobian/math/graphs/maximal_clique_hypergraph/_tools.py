"""Maximal-clique hypergraph operation declarations."""

from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationExample,
)
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphRequest,
    MaximalCliqueHypergraphResult,
)
from jacobian.math.graphs.maximal_clique_hypergraph.operations import (
    construct_maximal_clique_hypergraph,
)


def compute_maximal_clique_hypergraph(
    request: MaximalCliqueHypergraphRequest,
) -> MaximalCliqueHypergraphResult:
    return construct_maximal_clique_hypergraph(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.maximal_clique_hypergraph.construct",
        title="Construct the maximal-clique hypergraph of a graph",
        description=(
            "Construct a source-bound FiniteHypergraph whose vertices are "
            "the graph's vertices and whose hyperedges are the inclusion-"
            "maximal complete vertex sets of cardinality at least two "
            "(nontrivial maximal cliques). The complete family is returned "
            "in deterministic source-vertex order."
        ),
        request_type=MaximalCliqueHypergraphRequest,
        result_type=MaximalCliqueHypergraphResult,
        run=compute_maximal_clique_hypergraph,
        tags=("graph", "hypergraph", "exact"),
        examples=(
            OperationExample(
                name="triangle_with_pendant",
                description=(
                    "Triangle x-y-z with pendant vertex r attached to z. "
                    "Edge endpoints use lexicographic label order, independent "
                    "of the vertex list order."
                ),
                input={
                    "graph": {
                        "vertices": ["x", "y", "z", "r"],
                        "edges": [
                            ["x", "y"],
                            ["x", "z"],
                            ["y", "z"],
                            ["r", "z"],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
