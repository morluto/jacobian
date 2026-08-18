"""Normal-form operation declarations for issue #1893."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._normal_form_operations import (
    k_free_decomposition,
    normalize_positive_quadratic_radical,
    perfect_power_profile,
    squarefree_decomposition,
)
from jacobian.math.number_theory._normal_forms import (
    KFreeDecompositionRequest,
    KFreeDecompositionResult,
    MaximalPerfectPowerResult,
    NormalizedQuadraticRadicalResult,
    PerfectPowerProfileRequest,
    QuadraticRadicalNormalizeRequest,
    SquarefreeDecompositionRequest,
    SquarefreeDecompositionResult,
)
from jacobian.math.number_theory._support import number_theory_operation

NORMAL_FORM_OPERATIONS = (
    number_theory_operation(
        "integer.perfect_power.profile.compute",
        "Compute maximal perfect-power profile",
        (
            "Compute the maximal perfect-power profile of one integer: the "
            "canonical base and maximal exponent such that b^e = n, handling "
            "zero, units, and negative integers correctly."
        ),
        PerfectPowerProfileRequest,
        MaximalPerfectPowerResult,
        perfect_power_profile,
        "number-theory",
        "normal-form",
        examples=(
            example(
                "perfect_power_64",
                "Compute the maximal perfect-power profile of 64.",
                {"value": "64"},
            ),
        ),
    ),
    number_theory_operation(
        "integer.k_free_decomposition.compute",
        "Compute k-free decomposition",
        (
            "Decompose one integer n into a^k * c where a >= 1, c has the same "
            "sign as n, and no prime to the k-th power divides |c|."
        ),
        KFreeDecompositionRequest,
        KFreeDecompositionResult,
        k_free_decomposition,
        "number-theory",
        "normal-form",
        examples=(
            example(
                "k_free_72_k3",
                "Decompose 72 as a^3 * c.",
                {"value": "72", "k": 3},
            ),
        ),
    ),
    number_theory_operation(
        "integer.squarefree_decomposition.compute",
        "Compute squarefree decomposition",
        (
            "Decompose one integer n into s^2 * d where s >= 1, d has the same "
            "sign as n, and |d| is squarefree."
        ),
        SquarefreeDecompositionRequest,
        SquarefreeDecompositionResult,
        squarefree_decomposition,
        "number-theory",
        "normal-form",
        examples=(
            example(
                "squarefree_72",
                "Compute the squarefree decomposition of 72.",
                {"value": "72"},
            ),
        ),
    ),
    number_theory_operation(
        "quadratic_radical.positive_integer.normalize.compute",
        "Normalize positive quadratic radical",
        (
            "Normalize sqrt(n) = s * sqrt(d) for one nonnegative integer n, "
            "where s >= 0, d >= 1 is squarefree, and s^2 * d = n."
        ),
        QuadraticRadicalNormalizeRequest,
        NormalizedQuadraticRadicalResult,
        normalize_positive_quadratic_radical,
        "number-theory",
        "normal-form",
        examples=(
            example(
                "radical_72",
                "Normalize sqrt(72) = 6 * sqrt(2).",
                {"value": "72"},
            ),
        ),
    ),
)

__all__ = ["NORMAL_FORM_OPERATIONS"]
