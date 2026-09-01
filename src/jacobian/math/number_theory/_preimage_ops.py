"""Declarations for multiplier divisor-sum fibers and p-adic profiles."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory._preimage_models import (
    KSigmaPreimageRequest,
    KSigmaPreimageResult,
    PAdicIntervalProfileRequest,
    PAdicIntervalProfileResult,
)
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
    MathTool(
        operation_id="number_theory.ksigma.preimage.compute",
        title="Compute the preimage of k*sigma(n)",
        description="Find all positive n such that k * sigma(n) = target_value, where sigma is the sum-of-divisors function.",
        request_type=KSigmaPreimageRequest,
        result_type=KSigmaPreimageResult,
        run=compute_ksigma_preimage,
        tags=("number-theory", "divisor-function", "preimage", "exact"),
        examples=(
            OperationExample(
                name="ksigma_preimage_2_8",
                description="Find all n with 2*sigma(n) = 8.",
                input={"k": 2, "target_value": "8"},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.integer_interval.p_adic_valuation_profile.compute",
        title="Compute a p-adic valuation profile",
        description="For each valuation j, count the integers in [start+1, start+length] with v_p(n) = j.",
        request_type=PAdicIntervalProfileRequest,
        result_type=PAdicIntervalProfileResult,
        run=compute_p_adic_interval_profile,
        tags=("number-theory", "p-adic", "interval-profile", "exact"),
        examples=(
            OperationExample(
                name="p_adic_profile_2_0_10",
                description="Compute the valuation histogram for 1 through 10 at p=2; "
                "the coupled endpoint start + length and its exact sum, work, "
                "and canonical result all fit the admission envelope.",
                input={"start": "0", "length": "10", "prime": "2"},
            ),
        ),
    ),
)

__all__ = ["PREIMAGE_OPERATIONS"]
