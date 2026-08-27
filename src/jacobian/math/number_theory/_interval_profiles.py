"""Declarations for bounded integer-interval arithmetic-function profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._interval_profile_models import (
    DivisorCountProfileResult,
    GreatestPrimeFactorProfileResult,
    IntervalProfileRequest,
    PrimeGapProfileResult,
    SquarefreeProfileResult,
)
from jacobian.math.number_theory._interval_profile_operations import (
    compute_divisor_count_profile,
    compute_greatest_prime_factor_profile,
    compute_prime_gap_profile,
    compute_squarefree_profile,
)
from jacobian.math.number_theory._support import number_theory_operation

INTERVAL_PROFILE_OPERATIONS = (
    number_theory_operation(
        "number_theory.integer_interval.squarefree_profile.compute",
        "Compute squarefree profile on a bounded interval",
        (
            "Partition a closed positive integer interval [L, U] into its "
            "exact squarefree and non-squarefree members, retaining ordered "
            "lists and counts for both classes."
        ),
        IntervalProfileRequest,
        SquarefreeProfileResult,
        compute_squarefree_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "squarefree_interval_1_to_12",
                (
                    "Partition [1, 12] into squarefree and non-squarefree "
                    "integers; the lower bound must be at least 1 and the "
                    "upper bound must not be less than the lower bound."
                ),
                {"lower_bound": 1, "upper_bound": 12},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.divisor_count_profile.compute",
        "Compute divisor-count profile on a bounded interval",
        (
            "Return the complete ordered table (n, tau(n)) for every integer "
            "n in a closed positive interval [L, U], where tau(n) is the "
            "number of positive divisors of n."
        ),
        IntervalProfileRequest,
        DivisorCountProfileResult,
        compute_divisor_count_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "divisor_count_interval_1_to_12",
                (
                    "Compute tau(n) for each n from 1 to 12; the lower bound "
                    "must be at least 1 and the upper bound must not be less "
                    "than the lower bound."
                ),
                {"lower_bound": 1, "upper_bound": 12},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.greatest_prime_factor_profile.compute",
        "Compute greatest-prime-factor profile on a bounded interval",
        (
            "Return the complete ordered table (n, P+(n)) for every integer "
            "n in a closed positive interval [L, U], where P+(1) = 1 and "
            "P+(n) is the largest prime divisor of n for n >= 2."
        ),
        IntervalProfileRequest,
        GreatestPrimeFactorProfileResult,
        compute_greatest_prime_factor_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "gpf_interval_1_to_10",
                (
                    "Compute P+(n) for each n from 1 to 10; the lower bound "
                    "must be at least 1 and the upper bound must not be less "
                    "than the lower bound."
                ),
                {"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.prime_gap_profile.compute",
        "Compute consecutive-prime gap profile on a bounded interval",
        (
            "Return every consecutive-prime pair (p, q, q - p) for which the "
            "lower endpoint p lies in a closed positive interval [L, U], "
            "including the successor prime beyond U when needed to complete "
            "the last gap."
        ),
        IntervalProfileRequest,
        PrimeGapProfileResult,
        compute_prime_gap_profile,
        "number-theory",
        "prime",
        "interval-profile",
        examples=(
            example(
                "prime_gap_interval_3_to_5",
                (
                    "Compute consecutive-prime gaps for primes with lower "
                    "endpoint between 3 and 5; the lower bound must be at "
                    "least 1 and the upper bound must not be less than the "
                    "lower bound."
                ),
                {"lower_bound": 3, "upper_bound": 5},
            ),
        ),
    ),
)

__all__ = ["INTERVAL_PROFILE_OPERATIONS"]
