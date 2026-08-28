"""Domain adapter for graphical model operations."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.graphical_models._models import (
    DSeparationRequest,
    DSeparationResult,
    FactorMarginalizeRequest,
    FactorMarginalizeResult,
    FactorMultiplyRequest,
    FactorMultiplyResult,
)
from jacobian.math.probability.graphical_models._validation import (
    validate_d_separation_input,
)
from jacobian.math.probability.graphical_models.operations import (
    d_separation,
    factor_marginalize,
    factor_multiply,
)

__all__ = [
    "compute_d_separation",
    "compute_factor_marginalize",
    "compute_factor_multiply",
]


def compute_factor_multiply(request: FactorMultiplyRequest) -> FactorMultiplyResult:
    if request.left.domain_sizes != request.right.domain_sizes:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="graphical_model.factor_domains_mismatch",
            message="factors must share the exact model domain_sizes",
        )
    return FactorMultiplyResult._from_kernel(
        request.left, request.right, factor_multiply(request.left, request.right)
    )


def compute_factor_marginalize(
    request: FactorMarginalizeRequest,
) -> FactorMarginalizeResult:
    if request.variable not in request.factor.variables:
        raise OperationDomainValidationError(
            location=("variable",),
            code="graphical_model.factor_variable_missing",
            message="variable is not in factor",
        )
    return FactorMarginalizeResult._from_kernel(
        request.factor,
        request.variable,
        factor_marginalize(request.factor, request.variable),
    )


def compute_d_separation(request: DSeparationRequest) -> DSeparationResult:
    try:
        validate_d_separation_input(
            request.variable_count,
            request.edges,
            request.set_a,
            request.set_b,
            request.set_c,
        )
    except ValueError as error:
        raise OperationDomainValidationError(
            location=("edges", "set_a", "set_b", "set_c"),
            code="graphical_model.d_separation_invalid",
            message=str(error),
        ) from error
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
