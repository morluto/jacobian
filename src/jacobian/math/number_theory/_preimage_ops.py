"""Declarations for multiplier divisor-sum fibers and p-adic profiles."""

from jacobian.catalog._examples import example
from jacobian.math.number_theory._preimage_models import (
    KSigmaPreimageRequest,
    KSigmaPreimageResult,
    PAdicIntervalProfileRequest,
    PAdicIntervalProfileResult,
)
from jacobian.math.number_theory._support import number_theory_operation
from jacobian.math.number_theory.operations import (
    ksigma_preimage,
    p_adic_interval_profile,
)


def compute_ksigma_preimage(
    request: KSigmaPreimageRequest,
) -> KSigmaPreimageResult:
    """Project a wire request onto the canonical preimage operation."""

    return ksigma_preimage(
        request.k,
        int(request.target_value),
    )


def compute_p_adic_interval_profile(
    request: PAdicIntervalProfileRequest,
) -> PAdicIntervalProfileResult:
    """Project a wire request onto the canonical valuation operation."""

    return p_adic_interval_profile(
        int(request.start),
        int(request.length),
        int(request.prime),
    )


PREIMAGE_OPERATIONS = (
    number_theory_operation(
        "number_theory.ksigma.preimage.compute",
        "Compute the preimage of k*sigma(n)",
        "Find all positive n such that k * sigma(n) = target_value, where sigma is the sum-of-divisors function.",
        KSigmaPreimageRequest,
        KSigmaPreimageResult,
        compute_ksigma_preimage,
        "number-theory",
        "divisor-function",
        "preimage",
        "exact",
        examples=(
            example(
                "ksigma_preimage_2_8",
                "Find all n with 2*sigma(n) = 8.",
                {"k": 2, "target_value": "8"},
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
