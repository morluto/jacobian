"""Declarations for translated-prime representation profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._prime_shift_models import (
    MAX_SHIFT_RESULT_BYTES,
    MAX_SHIFT_WORK,
    PrimeShiftProfileRequest,
    PrimeShiftProfileResult,
)
from jacobian.math.number_theory._prime_shift_operations import (
    compute_prime_shift_profile,
)
from jacobian.math.number_theory._support import number_theory_operation

PRIME_SHIFT_OPERATION = number_theory_operation(
    "number_theory.translated_prime.representation_profile.compute",
    "Compute translated-prime representation profile on a bounded interval",
    (
        "For each n in [L, U], count representations n = p + 2^k where p is "
        "prime and k >= 0. Returns the complete ordered profile when its "
        f"canonical JSON is within the {MAX_SHIFT_RESULT_BYTES}-byte output "
        f"budget and its segmented-sieve work is within {MAX_SHIFT_WORK} "
        "units. Endpoint size is admitted by that derived work envelope rather "
        "than by a fixed scalar cap."
    ),
    PrimeShiftProfileRequest,
    PrimeShiftProfileResult,
    compute_prime_shift_profile,
    "number-theory",
    "prime",
    "interval-profile",
    examples=(
        example(
            "prime_shift_1_20",
            "Compute translated-prime representations for n from 1 to 20.",
            {"lower_bound": 1, "upper_bound": 20},
        ),
    ),
)

__all__ = ["PRIME_SHIFT_OPERATION"]
