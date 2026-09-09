"""Declarations for 3-term progression hypergraph construction."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.combinatorics._progression_hypergraph_models import (
    ProgressionHypergraphRequest,
    ProgressionHypergraphResult,
)
from jacobian.math.combinatorics.operations import progression_hypergraph


def construct_3term_progression_hypergraph(
    request: ProgressionHypergraphRequest,
) -> ProgressionHypergraphResult:
    return progression_hypergraph(request.group)


PROGRESSION_HYPERGRAPH_OPERATION = MathTool(
    operation_id="combinatorics.finite_abelian.3term_progression_hypergraph.construct",
    title="Construct a finite Abelian 3-term progression hypergraph",
    description=(
        "Construct the complete 3-uniform hypergraph whose edges are the "
        "nondegenerate 3-term progressions in an explicit product of cyclic groups."
    ),
    request_type=ProgressionHypergraphRequest,
    result_type=ProgressionHypergraphResult,
    run=construct_3term_progression_hypergraph,
    tags=("combinatorics", "additive-combinatorics", "hypergraph"),
    examples=(
        OperationExample(
            name="three_ap_z5",
            description="Construct the 3-AP hypergraph of Z/5Z.",
            input={"group": {"moduli": ["5"]}},
        ),
    ),
)

__all__ = ["PROGRESSION_HYPERGRAPH_OPERATION"]
