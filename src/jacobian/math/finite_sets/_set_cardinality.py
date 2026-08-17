"""Finite-set cardinality operations."""

from jacobian.catalog._examples import example
from jacobian.math.finite_sets._models import (
    FiniteSetCardinalityResult,
    FiniteSetPairRequest,
)
from jacobian.math.finite_sets._operations import (
    intersection_cardinality,
    left_cardinality,
    union_cardinality,
)
from jacobian.math.finite_sets._support import finite_set_operation

SET_CARDINALITY_OPERATIONS = (
    finite_set_operation(
        "finite_set.compute.left_cardinality",
        "Count left finite set",
        "Count distinct elements in the left finite integer set.",
        FiniteSetPairRequest,
        FiniteSetCardinalityResult,
        left_cardinality,
        "finite-set",
        "counting",
        examples=(
            example(
                "left_cardinality_1_2_3",
                "Count elements in the left finite set.",
                {"left": {"elements": ["1", "2", "3"]}, "right": {"elements": ["2"]}},
            ),
        ),
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
        examples=(
            example(
                "intersection_cardinality_1_2_and_2_3",
                "Count common elements of two finite sets.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
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
        examples=(
            example(
                "union_cardinality_1_2_and_2_3",
                "Count elements in the union of two finite sets.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
    ),
)
