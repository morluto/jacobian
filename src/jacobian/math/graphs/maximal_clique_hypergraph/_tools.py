"""Maximal-clique hypergraph operation declarations."""

from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.catalog.models import (
    MathTool,
    MathTools,
    OperationDomainValidationError,
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
    result = construct_maximal_clique_hypergraph(request.graph)
    try:
        encode_strict_json(result.model_dump(mode="json"))
    except CanonicalizationError as error:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph.maximal_clique_hypergraph.output_bound",
            message=(
                "the source-bound maximal-clique hypergraph result exceeds the "
                f"{CanonicalLimits().max_output_bytes}-byte canonical output bound"
            ),
        ) from error
    return result


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
                description="Triangle 0-1-2 with pendant vertex 3 attached to 2.",
                input={
                    "graph": {
                        "vertices": ["0", "1", "2", "3"],
                        "edges": [
                            ["0", "1"],
                            ["0", "2"],
                            ["1", "2"],
                            ["2", "3"],
                        ],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
