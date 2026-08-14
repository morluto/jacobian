"""Finite-set predicate operations."""

from jacobian.contracts.finite_sets import (
    FiniteSetBooleanResult,
    FiniteSetCoverageRequest,
    FiniteSetCoverageResult,
    FiniteSetPairRequest,
)
from jacobian.domains._examples import example
from jacobian.domains.finite_sets._support import finite_set_operation
from jacobian.domains.finite_sets.operations import (
    decide_disjoint,
    decide_exact_cover,
    decide_proper_subset,
    decide_subset,
)

SET_PREDICATE_OPERATIONS = (
    finite_set_operation(
        "finite_set.decide.exact_cover",
        "Decide exact finite-set coverage",
        "Decide whether a bounded sequence contains every scope element exactly once.",
        FiniteSetCoverageRequest,
        FiniteSetCoverageResult,
        decide_exact_cover,
        "finite-set",
        "predicate",
        examples=(
            example(
                "cover_1_2_3_once",
                "Check an exactly-once cover of a finite integer set.",
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
        examples=(
            example(
                "subset_1_2_of_1_2_3",
                "Check a finite-set subset relation.",
                {
                    "left": {"elements": ["1", "2"]},
                    "right": {"elements": ["1", "2", "3"]},
                },
            ),
        ),
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
            example(
                "proper_subset_1_2_of_1_2_3",
                "Check a proper finite-set subset relation.",
                {
                    "left": {"elements": ["1", "2"]},
                    "right": {"elements": ["1", "2", "3"]},
                },
            ),
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
        examples=(
            example(
                "disjoint_1_2_and_3_4",
                "Check whether two finite sets are disjoint.",
                {"left": {"elements": ["1", "2"]}, "right": {"elements": ["3", "4"]}},
            ),
        ),
    ),
)
