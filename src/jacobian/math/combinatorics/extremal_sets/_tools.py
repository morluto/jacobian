"""Binary-union relation hypergraph operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.extremal_sets._models import (
    BinaryUnionRelationRequest,
    BinaryUnionRelationResult,
    SunflowerHypergraphRequest,
    SunflowerHypergraphResult,
)
from jacobian.math.combinatorics.extremal_sets._sunflower_r import (
    SunflowerFamilyRequest,
    SunflowerFamilyResult,
    construct_sunflower_family,
)
from jacobian.math.combinatorics.extremal_sets.operations import (
    construct_binary_union_relation,
    construct_sunflower_hypergraph,
)


def compute_sunflower_family(
    request: SunflowerFamilyRequest,
) -> SunflowerFamilyResult:
    return construct_sunflower_family(request)


def compute_binary_union_relation(
    request: BinaryUnionRelationRequest,
) -> BinaryUnionRelationResult:
    return construct_binary_union_relation(request.source)


TOOLS: MathTools = (
    MathTool(
        operation_id="set_system.sunflower_family.construct",
        title="Construct the complete sunflower family for a petal count",
        description=(
            "Return every distinct-index subfamily of exactly r >= 2 members "
            "whose pairwise intersections are all equal to one common core, "
            "together with each row's exact core, the derived count, "
            "sunflower-free status, and a canonical r-uniform hypergraph "
            "edge projection. The "
            "complete accepted construction returns every row or fails before "
            "expansion."
        ),
        request_type=SunflowerFamilyRequest,
        result_type=SunflowerFamilyResult,
        run=compute_sunflower_family,
        tags=("combinatorics", "set-system", "sunflower", "hypergraph", "complete"),
        examples=(
            OperationExample(
                name="four_petals",
                description="Four petals sharing the core {0} form one sunflower.",
                input={
                    "source": {
                        "ground_set_size": 5,
                        "members": [[0, 1], [0, 2], [0, 3], [0, 4]],
                    },
                    "petal_count": 4,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="set_system.sunflower_triple_hypergraph.construct",
        title="Construct the complete sunflower-triple hypergraph",
        description=(
            "Return every three-member subfamily whose three pairwise intersections "
            "are equal. Each hyperedge retains its exact common core and source indices."
        ),
        request_type=SunflowerHypergraphRequest,
        result_type=SunflowerHypergraphResult,
        run=lambda request: construct_sunflower_hypergraph(request.source),
        tags=("combinatorics", "set-system", "sunflower", "hypergraph", "complete"),
        examples=(
            OperationExample(
                name="three_petals",
                description="Construct one sunflower with core {0} and three petals.",
                input={
                    "source": {
                        "ground_set_size": 4,
                        "members": [[0, 1], [0, 2], [0, 3]],
                    }
                },
            ),
        ),
    ),
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
