"""Binary set-operation operations over finite integer sets."""

from jacobian.catalog._examples import example
from jacobian.math.combinatorics.finite_structures.sets._models import (
    FiniteSetElementListResult,
    FiniteSetPairRequest,
)
from jacobian.math.combinatorics.finite_structures.sets._operations import (
    set_difference,
    set_intersection,
    set_symmetric_difference,
    set_union,
)
from jacobian.math.combinatorics.finite_structures.sets._support import (
    finite_set_operation,
)

SET_OPERATION_OPERATIONS = (
    finite_set_operation(
        "finite_set.compute.union",
        "Compute finite-set union",
        "Return the sorted union of two finite integer sets.",
        FiniteSetPairRequest,
        FiniteSetElementListResult,
        set_union,
        "finite-set",
        "exact",
        examples=(
            example(
                "union_1_2_and_2_3",
                "Union two overlapping finite sets.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
            ),
            example(
                "union_disjoint_sets",
                "Compute the union of two disjoint sets; each set's elements must be unique.",
                {"left": {"elements": ["1", "5"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
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
        examples=(
            example(
                "intersection_1_2_and_2_3",
                "Intersect two finite sets.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
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
        examples=(
            example(
                "difference_1_2_minus_2_3",
                "Subtract one finite set from another.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
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
            example(
                "symmetric_difference",
                "Compute the symmetric difference of two finite sets.",
                {"left": {"elements": ["3", "1"]}, "right": {"elements": ["2", "3"]}},
            ),
        ),
    ),
)
