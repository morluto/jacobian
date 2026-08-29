"""Typed declarations for the binary-union relation hypergraph."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, MathTools
from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph._models import (
    BinaryUnionHypergraphRequest,
    BinaryUnionHypergraphResult,
)
from jacobian.math.combinatorics.extremal_sets.binary_union_hypergraph.operations import (
    compute_binary_union_hypergraph,
)


def _compute(request: BinaryUnionHypergraphRequest) -> BinaryUnionHypergraphResult:
    return compute_binary_union_hypergraph(request.sets)


TOOLS: MathTools = (
    MathTool(
        operation_id="set_system.binary_union_relation_hypergraph.compute",
        title="Construct exact binary-union relation hypergraphs",
        description=(
            "Given a finite indexed family of distinct finite sets, return the "
            "complete canonical 3-uniform relation hypergraph whose edge "
            "{i,j,k} records S_i union S_j = S_k among three distinct source members."
        ),
        request_type=BinaryUnionHypergraphRequest,
        result_type=BinaryUnionHypergraphResult,
        run=_compute,
        tags=("set_system", "union", "hypergraph", "exact"),
        examples=(
            example(
                "simple",
                "Binary union relations among {1}, {2}, {1,2}.",
                {"sets": [[1], [2], [1, 2]]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
