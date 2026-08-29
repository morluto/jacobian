"""Typed declarations for the maximal-clique hypergraph operation."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.graphs.maximal_clique_hypergraph._models import (
    MaximalCliqueHypergraphRequest,
    MaximalCliqueHypergraphResult,
)
from jacobian.math.graphs.maximal_clique_hypergraph.operations import (
    construct_maximal_clique_hypergraph,
)


def _construct(
    request: MaximalCliqueHypergraphRequest,
) -> MaximalCliqueHypergraphResult:
    return construct_maximal_clique_hypergraph(request.graph)


TOOLS: MathTools = (
    MathTool(
        operation_id="graph.maximal_clique_hypergraph.construct",
        title="Construct the maximal-clique hypergraph of a graph",
        description=(
            "For one bounded simple undirected graph, return a canonical "
            "FiniteHypergraph whose hyperedges are exactly the inclusion-maximal "
            "cliques of cardinality at least two."
        ),
        request_type=MaximalCliqueHypergraphRequest,
        result_type=MaximalCliqueHypergraphResult,
        run=_construct,
        tags=("graph", "clique", "hypergraph", "exact"),
        examples=(
            example(
                "triangle",
                "The maximal clique hypergraph of a triangle.",
                {
                    "graph": {
                        "vertices": ["a", "b", "c"],
                        "edges": [["a", "b"], ["b", "c"], ["a", "c"]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
