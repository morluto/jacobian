"""Domain adapter for graphical model operations."""

from __future__ import annotations

from jacobian.math.graphical_models._models import (
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMarginalizeResult,
    FactorMultiplyRequest,
    FactorMultiplyResult,
)
from jacobian.math.graphical_models.operations import (
    d_separation,
    factor_marginalize,
    factor_multiply,
)

__all__ = [
    "compute_d_separation",
    "compute_factor_marginalize",
    "compute_factor_multiply",
    "verify_d_separation_result",
    "verify_factor_marginalize_result",
    "verify_factor_multiply_result",
]


def compute_factor_multiply(request: FactorMultiplyRequest) -> FactorMultiplyResult:
    return FactorMultiplyResult._from_kernel(
        request.left, request.right, factor_multiply(request.left, request.right)
    )


def compute_factor_marginalize(
    request: FactorMarginalizeRequest,
) -> FactorMarginalizeResult:
    return FactorMarginalizeResult._from_kernel(
        request.factor,
        request.variable,
        factor_marginalize(request.factor, request.variable),
    )


def compute_d_separation(request: DSeparationRequest) -> DSeparationResult:
    return DSeparationResult._from_kernel(
        request,
        d_separation(
            request.variable_count,
            request.edges,
            request.set_a,
            request.set_b,
            request.set_c,
        ),
    )


def verify_factor_multiply_result(
    request: FactorMultiplyRequest, result: FactorMultiplyResult
) -> bool:
    """Verify an independently supplied factor-product claim."""

    return (
        result.left == request.left
        and result.right == request.right
        and result.factor == factor_multiply(request.left, request.right)
    )


def verify_factor_marginalize_result(
    request: FactorMarginalizeRequest, result: FactorMarginalizeResult
) -> bool:
    """Verify an independently supplied bounded factor marginal."""

    return (
        result.source_factor == request.factor
        and result.variable == request.variable
        and result.factor == factor_marginalize(request.factor, request.variable)
    )


def verify_d_separation_result(
    request: DSeparationRequest, result: DSeparationResult
) -> bool:
    """Verify one bounded d-separation decision supplied independently."""

    return (
        result.variable_count == request.variable_count
        and result.edges == request.edges
        and result.set_a == request.set_a
        and result.set_b == request.set_b
        and result.set_c == request.set_c
        and result.d_separated
        == d_separation(
            request.variable_count,
            request.edges,
            request.set_a,
            request.set_b,
            request.set_c,
        )
    )
