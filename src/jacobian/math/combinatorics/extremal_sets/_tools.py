"""Binary-union relation hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationRequest,
    BinaryUnionRelationResult,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_binary_union_relation,
)


def compute_binary_union_relation(
    request: BinaryUnionRelationRequest,
) -> BinaryUnionRelationResult:
    return construct_binary_union_relation(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="set_system.binary_union_relation_hypergraph.compute",
        title="Compute the binary-union relation hypergraph of a set family",
        description=(
            "Given a declared finite ground-set axis and an indexed family of "
            "distinct subsets, return every distinct-index equation S_i union "
            "S_j = S_k. Rows retain the operand/result orientation, and each "
            "row is bound by ID to its edge in the 3-uniform hypergraph projection."
        ),
        request_type=BinaryUnionRelationRequest,
        result_type=BinaryUnionRelationResult,
        run=compute_binary_union_relation,
        tags=("combinatorics", "extremal-set-theory", "exact"),
        examples=(
            OperationExample(
                name="boolean_lattice_2",
                description="Family {empty, {a}, {b}, {a,b}} has one union relation.",
                input={
                    "source": {
                        "ground_set_size": 2,
                        "members": [[], [0], [1], [0, 1]],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
