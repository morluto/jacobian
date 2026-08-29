"""Finite-integer-set operation declarations."""

from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTools
from jacobian.math.combinatorics.finite_structures.sets import operations as native
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteSetBooleanResult,
    FiniteSetCardinalityResult,
    FiniteSetCoverageRequest,
    FiniteSetCoverageResult,
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)
from jacobian.math.combinatorics.finite_structures.sets._support import (
    finite_set_operation,
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
    finite_set_operation(
        "finite_set.compute.union",
        "Compute finite-set union",
        "Return the sorted union of two finite integer sets.",
        FiniteSetPairRequest,
        FiniteSetElementListResult,
        set_union,
        "finite-set",
        "exact",
        examples=(example("overlapping_sets", "Union two finite sets.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.intersection",
        "Compute finite-set intersection",
        "Return the sorted intersection of two finite integer sets.",
        FiniteSetPairRequest,
        FiniteSetElementListResult,
        set_intersection,
        "finite-set",
        "exact",
        examples=(example("overlapping_sets", "Intersect two finite sets.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.difference",
        "Compute finite-set difference",
        "Return elements in the first finite set but not the second.",
        FiniteSetPairRequest,
        FiniteSetElementListResult,
        set_difference,
        "finite-set",
        "exact",
        examples=(example("overlapping_sets", "Subtract two finite sets.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.symmetric_difference",
        "Compute symmetric difference",
        "Return elements occurring in exactly one of two finite integer sets.",
        FiniteSetPairRequest,
        FiniteSetElementListResult,
        set_symmetric_difference,
        "finite-set",
        "exact",
        examples=(
            example("overlapping_sets", "Compute the symmetric difference.", _PAIR),
        ),
    ),
    finite_set_operation(
        "finite_set.decide.exact_cover",
        "Decide exact finite-set coverage",
        "Decide whether a sequence contains every scope element exactly once.",
        FiniteSetCoverageRequest,
        FiniteSetCoverageResult,
        decide_exact_cover,
        "finite-set",
        "predicate",
        examples=(
            example(
                "cover_once",
                "Check an exactly-once cover.",
                {"scope": {"elements": ["1", "2", "3"]}, "values": ["3", "1", "2"]},
            ),
        ),
    ),
    finite_set_operation(
        "finite_set.decide.subset",
        "Decide subset relation",
        "Decide whether every left-set element occurs in the right set.",
        FiniteSetPairRequest,
        FiniteSetBooleanResult,
        decide_subset,
        "finite-set",
        "predicate",
        examples=(example("overlapping_sets", "Check the subset relation.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.decide.proper_subset",
        "Decide proper subset",
        "Decide whether the left set is a strict subset of the right set.",
        FiniteSetPairRequest,
        FiniteSetBooleanResult,
        decide_proper_subset,
        "finite-set",
        "predicate",
        examples=(
            example("overlapping_sets", "Check the proper-subset relation.", _PAIR),
        ),
    ),
    finite_set_operation(
        "finite_set.decide.disjoint",
        "Decide disjointness",
        "Decide whether two finite integer sets have empty intersection.",
        FiniteSetPairRequest,
        FiniteSetBooleanResult,
        decide_disjoint,
        "finite-set",
        "predicate",
        examples=(example("overlapping_sets", "Check disjointness.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.left_cardinality",
        "Count left finite set",
        "Count distinct elements in the left finite integer set.",
        FiniteSetPairRequest,
        FiniteSetCardinalityResult,
        left_cardinality,
        "finite-set",
        "counting",
        examples=(example("overlapping_sets", "Count the left set.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.intersection_cardinality",
        "Count set intersection",
        "Count common elements of two finite integer sets.",
        FiniteSetPairRequest,
        FiniteSetCardinalityResult,
        intersection_cardinality,
        "finite-set",
        "counting",
        examples=(example("overlapping_sets", "Count the intersection.", _PAIR),),
    ),
    finite_set_operation(
        "finite_set.compute.union_cardinality",
        "Count set union",
        "Count distinct elements occurring in either finite integer set.",
        FiniteSetPairRequest,
        FiniteSetCardinalityResult,
        union_cardinality,
        "finite-set",
        "counting",
        examples=(example("overlapping_sets", "Count the union.", _PAIR),),
    ),
)

__all__ = ["TOOLS"]
