"""Finite-integer-set operation declarations."""

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.combinatorics.finite_structures.sets import operations as native
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteSetBooleanResult,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetCoverageResult,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)


def decide_exact_cover(request: FiniteSetCoverageRequest) -> FiniteSetCoverageResult:
    return native.exact_cover(request.scope, request.values)


def set_union(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_union(request.left, request.right).elements
    )


def set_intersection(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_intersection(request.left, request.right).elements
    )


def set_difference(request: FiniteSetPairRequest) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_difference(request.left, request.right).elements
    )


def set_symmetric_difference(
    request: FiniteSetPairRequest,
) -> FiniteSetElementListResult:
    return FiniteSetElementListResult(
        elements=native.set_symmetric_difference(request.left, request.right).elements
    )


def decide_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(holds=native.is_subset(request.left, request.right))


def decide_proper_subset(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(
        holds=native.is_proper_subset(request.left, request.right)
    )


def decide_disjoint(request: FiniteSetPairRequest) -> FiniteSetBooleanResult:
    return FiniteSetBooleanResult(
        holds=native.are_disjoint(request.left, request.right)
    )


def left_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(cardinality=native.cardinality(request.left))


def intersection_cardinality(
    request: FiniteSetPairRequest,
) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(
        cardinality=native.intersection_cardinality(request.left, request.right)
    )


def union_cardinality(request: FiniteSetPairRequest) -> FiniteSetCardinalityResult:
    return FiniteSetCardinalityResult(
        cardinality=native.union_cardinality(request.left, request.right)
    )


_PAIR = {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}}

TOOLS: MathTools = (
    MathTool(
        operation_id="finite_set.compute.union",
        title="Compute finite-set union",
        description="Return the sorted union of two finite integer sets.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetElementListResult,
        run=set_union,
        tags=("finite-set", "exact"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Union two finite sets.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.intersection",
        title="Compute finite-set intersection",
        description="Return the sorted intersection of two finite integer sets.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetElementListResult,
        run=set_intersection,
        tags=("finite-set", "exact"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Intersect two finite sets.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.difference",
        title="Compute finite-set difference",
        description="Return elements in the first finite set but not the second.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetElementListResult,
        run=set_difference,
        tags=("finite-set", "exact"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Subtract two finite sets.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.symmetric_difference",
        title="Compute symmetric difference",
        description="Return elements occurring in exactly one of two finite integer sets.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetElementListResult,
        run=set_symmetric_difference,
        tags=("finite-set", "exact"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Compute the symmetric difference.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.decide.exact_cover",
        title="Decide exact finite-set coverage",
        description="Decide whether a sequence contains every scope element exactly once.",
        request_type=FiniteSetCoverageRequest,
        result_type=FiniteSetCoverageResult,
        run=decide_exact_cover,
        tags=("finite-set", "predicate"),
        examples=(
            OperationExample(
                name="cover_once",
                description="Check an exactly-once cover.",
                input={
                    "scope": {"elements": ["1", "2", "3"]},
                    "values": ["3", "1", "2"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.decide.subset",
        title="Decide subset relation",
        description="Decide whether every left-set element occurs in the right set.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetBooleanResult,
        run=decide_subset,
        tags=("finite-set", "predicate"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Check the subset relation.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.decide.proper_subset",
        title="Decide proper subset",
        description="Decide whether the left set is a strict subset of the right set.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetBooleanResult,
        run=decide_proper_subset,
        tags=("finite-set", "predicate"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Check the proper-subset relation.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.decide.disjoint",
        title="Decide disjointness",
        description="Decide whether two finite integer sets have empty intersection.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetBooleanResult,
        run=decide_disjoint,
        tags=("finite-set", "predicate"),
        examples=(
            OperationExample(
                name="overlapping_sets", description="Check disjointness.", input=_PAIR
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.left_cardinality",
        title="Count left finite set",
        description="Count distinct elements in the left finite integer set.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetCardinalityResult,
        run=left_cardinality,
        tags=("finite-set", "counting"),
        examples=(
            OperationExample(
                name="overlapping_sets", description="Count the left set.", input=_PAIR
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.intersection_cardinality",
        title="Count set intersection",
        description="Count common elements of two finite integer sets.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetCardinalityResult,
        run=intersection_cardinality,
        tags=("finite-set", "counting"),
        examples=(
            OperationExample(
                name="overlapping_sets",
                description="Count the intersection.",
                input=_PAIR,
            ),
        ),
    ),
    MathTool(
        operation_id="finite_set.compute.union_cardinality",
        title="Count set union",
        description="Count distinct elements occurring in either finite integer set.",
        request_type=FiniteSetPairRequest,
        result_type=FiniteSetCardinalityResult,
        run=union_cardinality,
        tags=("finite-set", "counting"),
        examples=(
            OperationExample(
                name="overlapping_sets", description="Count the union.", input=_PAIR
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
