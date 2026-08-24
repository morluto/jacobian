"""Declarations for finite periodic congruence unions."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._periodic_models import (
    PeriodicCongruenceUnionMeasureResult,
    PeriodicCongruenceUnionProfileRequest,
    PeriodicCongruenceUnionProfileResult,
    PeriodicCongruenceUnionRequest,
)
from jacobian.math.number_theory._periodic_operations import (
    compute_periodic_congruence_union_measure,
    compute_periodic_congruence_union_profile,
)
from jacobian.math.number_theory._support import number_theory_operation

PERIODIC_CONGRUENCE_OPERATIONS = (
    number_theory_operation(
        "congruence.periodic_union.measure.compute",
        "Measure finite periodic congruence union",
        (
            "Compute the exact occupied count and rational density of a normalized "
            "finite union of residue subsets in the least common multiple of their "
            "moduli, optionally complemented, without requiring residue materialization."
        ),
        PeriodicCongruenceUnionRequest,
        PeriodicCongruenceUnionMeasureResult,
        compute_periodic_congruence_union_measure,
        "number-theory",
        "combinatorics",
        "congruence",
        "periodic",
        "density",
        examples=(
            example(
                "overlapping_periodic_union_measure",
                (
                    "Measure the union of residues {0,1} modulo 4 and {-1,1} "
                    "modulo 6; moduli must be positive decimal strings and residue "
                    "representatives are normalized modulo their modulus."
                ),
                {
                    "subsets": [
                        {"modulus": "4", "residues": ["0", "1"]},
                        {"modulus": "6", "residues": ["-1", "1"]},
                    ],
                    "complement": False,
                },
            ),
        ),
    ),
    number_theory_operation(
        "congruence.periodic_union.profile.compute",
        "Materialize finite periodic congruence union",
        (
            "Return every occupied residue in the least common multiple period of "
            "a normalized finite union of residue subsets, together with its exact "
            "count and density, optionally after complementing the union."
        ),
        PeriodicCongruenceUnionProfileRequest,
        PeriodicCongruenceUnionProfileResult,
        compute_periodic_congruence_union_profile,
        "number-theory",
        "combinatorics",
        "congruence",
        "periodic",
        "enumeration",
        examples=(
            example(
                "complemented_periodic_union_profile",
                (
                    "Materialize residues avoiding the even classes modulo 4 and "
                    "the class 1 modulo 3; the lcm period and complete residue output "
                    "must fit the materialization bounds."
                ),
                {
                    "subsets": [
                        {"modulus": "4", "residues": ["0", "2"]},
                        {"modulus": "3", "residues": ["1"]},
                    ],
                    "complement": True,
                },
            ),
        ),
    ),
)

__all__ = ["PERIODIC_CONGRUENCE_OPERATIONS"]
