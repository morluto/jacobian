"""Declarations for prime-coverage and binomial-valuation profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._binomial_valuation_models import (
    BinomialValuationProfileRequest,
    BinomialValuationProfileResult,
)
from jacobian.math.number_theory._prime_coverage_models import (
    PrimeCoverageProfileRequest,
    PrimeCoverageProfileResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import (
    binomial_valuation_profile,
    prime_coverage_profile,
)


def compute_prime_coverage_profile(
    request: PrimeCoverageProfileRequest,
) -> PrimeCoverageProfileResult:
    """Project a wire request into the canonical coverage operation."""
    return prime_coverage_profile(request.lower_bound, request.upper_bound)


def compute_binomial_valuation_profile(
    request: BinomialValuationProfileRequest,
) -> BinomialValuationProfileResult:
    """Project a wire request into the canonical binomial operation."""
    return binomial_valuation_profile(request.n, request.prime)


ADDITIONAL_NT_OPERATIONS = (
    number_theory_operation(
        "number_theory.integer_interval.prime_coverage_profile.compute",
        "Compute prime-coverage profile on a bounded interval",
        "Return the complete ordered table (n, omega(n)) for every n in [L, U], where omega(n) is the number of distinct prime factors.",
        PrimeCoverageProfileRequest,
        PrimeCoverageProfileResult,
        compute_prime_coverage_profile,
        "number-theory",
        "arithmetic-function",
        "interval-profile",
        examples=(
            example(
                "prime_coverage_1_10",
                "Compute omega(n) for each n from 1 to 10.",
                {"lower_bound": 1, "upper_bound": 10},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.binomial_valuation.profile.compute",
        "Compute p-adic valuation profile of binomial coefficients",
        "For a given n and prime p, return v_p(C(n,k)) for all k from 0 to n using Kummer's theorem.",
        BinomialValuationProfileRequest,
        BinomialValuationProfileResult,
        compute_binomial_valuation_profile,
        "number-theory",
        "p-adic",
        "binomial",
        examples=(
            example(
                "binomial_val_n10_p2",
                "Compute v_2(C(10,k)) for k=0..10; the prime must be a prime number.",
                {"n": 10, "prime": 2},
            ),
        ),
    ),
)

__all__ = ["ADDITIONAL_NT_OPERATIONS"]
