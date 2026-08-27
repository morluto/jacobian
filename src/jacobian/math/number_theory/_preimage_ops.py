"""Declarations for divisor-sum-product fibers and p-adic profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._preimage_models import (
    DivisorSumProductPreimageRequest,
    DivisorSumProductPreimageResult,
    PAdicIntervalProfileRequest,
    PAdicIntervalProfileResult,
)
from jacobian.math.number_theory._preimage_operations import (
    compute_divisor_sum_product_preimage,
    compute_p_adic_interval_profile,
)
from jacobian.math.number_theory._support import number_theory_operation

PREIMAGE_OPERATIONS = (
    number_theory_operation(
        "number_theory.divisor_sum_product.preimages.compute",
        "Compute the preimage of n*sigma(n)",
        "Find all positive n such that n * sigma(n) = target, where sigma is the sum-of-divisors function.",
        DivisorSumProductPreimageRequest,
        DivisorSumProductPreimageResult,
        compute_divisor_sum_product_preimage,
        "number-theory",
        "divisor-function",
        "preimage",
        "exact",
        examples=(
            example(
                "divisor_sum_product_preimage_336",
                "Find all n with n*sigma(n) = 336.",
                {"target": "336"},
            ),
        ),
    ),
    number_theory_operation(
        "number_theory.integer_interval.p_adic_valuation_profile.compute",
        "Compute a p-adic valuation profile",
        "For each valuation j, count the integers in [start+1, start+length] with v_p(n) = j.",
        PAdicIntervalProfileRequest,
        PAdicIntervalProfileResult,
        compute_p_adic_interval_profile,
        "number-theory",
        "p-adic",
        "interval-profile",
        "exact",
        examples=(
            example(
                "p_adic_profile_2_0_10",
                "Compute the valuation histogram for 1 through 10 at p=2; "
                "the coupled endpoint start + length and its exact sum, work, "
                "and canonical result all fit the admission envelope.",
                {"start": "0", "length": "10", "prime": "2"},
            ),
        ),
    ),
)

__all__ = ["PREIMAGE_OPERATIONS"]
