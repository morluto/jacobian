"""Declarations for bounded integer-interval arithmetic-function profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._interval_profile_models import (
    DivisorCountProfileRequest,
    DivisorCountProfileResult,
    DivisorSumProfileRequest,
    DivisorSumProfileResult,
    EulerTotientProfileRequest,
    EulerTotientProfileResult,
    GreatestPrimeFactorProfileRequest,
    GreatestPrimeFactorProfileResult,
    LeastPrimeFactorProfileRequest,
    LeastPrimeFactorProfileResult,
    PrimeGapProfileRequest,
    PrimeGapProfileResult,
    SquarefreeProfileRequest,
    SquarefreeProfileResult,
)
from jacobian.math.number_theory._interval_profile_operations import (
    compute_divisor_count_profile,
    compute_divisor_sum_profile,
    compute_euler_totient_profile,
    compute_greatest_prime_factor_profile,
    compute_least_prime_factor_profile,
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
        SquarefreeProfileRequest,
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
                    "integers; coupled width and result-size limits are "
                    "published in the request schema."
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
        DivisorCountProfileRequest,
        DivisorCountProfileResult,
        compute_divisor_count_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "divisor_count_interval_1_to_12",
                (
                    "Compute tau(n) for each n from 1 to 12; coupled width and "
                    "result-size limits are published in the request schema."
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
        GreatestPrimeFactorProfileRequest,
        GreatestPrimeFactorProfileResult,
        compute_greatest_prime_factor_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "gpf_interval_1_to_10",
                (
                    "Compute P+(n) for each n from 1 to 10; coupled width and "
                    "result-size limits are published in the request schema."
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
        PrimeGapProfileRequest,
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
                    "endpoint between 3 and 5; coupled width and result-size "
                    "limits are published in the request schema."
                ),
                {"lower_bound": 3, "upper_bound": 5},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.least_prime_factor_profile.compute",
        "Compute least-prime-factor profile on a bounded interval",
        "Return the complete ordered table (n, p(n)) for every n in [L, U], with p(1)=1.",
        LeastPrimeFactorProfileRequest,
        LeastPrimeFactorProfileResult,
        compute_least_prime_factor_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "lpf_1_10",
                "Compute p(n) for each n from 1 to 10.",
                {"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.euler_totient_profile.compute",
        "Compute Euler-totient profile on a bounded interval",
        "Return the complete ordered table (n, phi(n)) for every n in [L, U], with phi(1)=1.",
        EulerTotientProfileRequest,
        EulerTotientProfileResult,
        compute_euler_totient_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "totient_1_10",
                "Compute phi(n) for each n from 1 to 10.",
                {"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.divisor_sum_profile.compute",
        "Compute divisor-sum profile on a bounded interval",
        "Return the complete ordered table (n, sigma(n)) for every n in [L, U], with sigma(1)=1.",
        DivisorSumProfileRequest,
        DivisorSumProfileResult,
        compute_divisor_sum_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "divisor_sum_1_6",
                "Compute sigma(n) for each n from 1 to 6.",
                {"lower_bound": 1, "upper_bound": 6},
            ),
        ),
    ),
)

__all__ = ["INTERVAL_PROFILE_OPERATIONS"]
