"""Declarations for k*sigma preimage and p-adic interval valuation profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._preimage_models import (
    IntervalValuationProfileRequest,
    IntervalValuationProfileResult,
    KSigmaPreimageRequest,
    KSigmaPreimageResult,
)
from jacobian.math.number_theory._preimage_operations import (
    compute_interval_valuation_profile,
    compute_ksigma_preimage,
)
from jacobian.math.number_theory._support import number_theory_operation

PREIMAGE_OPERATIONS = (
    number_theory_operation(
        "number_theory.ksigma.preimage.compute",
        "Compute preimages of k*sigma(n)",
        "Find all n such that k * sigma(n) = target_value, where sigma is the sum-of-divisors function.",
        KSigmaPreimageRequest,
        KSigmaPreimageResult,
        compute_ksigma_preimage,
        "number-theory",
        "divisor-function",
        "preimage",
        examples=(
            example(
                "ksigma_preimage_basic",
                "Find n with 1*sigma(n) = 12.",
                {"k": 1, "target_value": 12},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.p_adic_valuation_profile.compute",
        "Compute p-adic valuation profile on a bounded interval",
        "For each n in [L, U], compute v_p(n) the p-adic valuation (largest power of p dividing n).",
        IntervalValuationProfileRequest,
        IntervalValuationProfileResult,
        compute_interval_valuation_profile,
        "number-theory",
        "p-adic",
        "interval-profile",
        examples=(
            example(
                "valuation_2_1_10",
                "Compute v_2(n) for n from 1 to 10.",
                {"lower_bound": 1, "upper_bound": 10, "prime": 2},
            ),
        ),
    ),
)

__all__ = ["PREIMAGE_OPERATIONS"]
