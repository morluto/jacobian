"""Catalog projections for exact subsystem-aware matrix operations."""

from jacobian.math.matrices.subsystems._models import (
    PsdOrderRequest,
    PsdOrderResult,
    SubsystemKroneckerProductRequest,
    SubsystemKroneckerProductResult,
    SubsystemPartialTraceRequest,
    SubsystemPartialTraceResult,
)
from jacobian.math.matrices.subsystems.operations import (
    kronecker_product,
    partial_trace,
    psd_order,
)


def compute_kronecker_product(
    request: SubsystemKroneckerProductRequest,
) -> SubsystemKroneckerProductResult:
    return SubsystemKroneckerProductResult(
        product=kronecker_product(request.left, request.right)
    )


def compute_partial_trace(
    request: SubsystemPartialTraceRequest,
) -> SubsystemPartialTraceResult:
    return SubsystemPartialTraceResult(
        source_matrix=request.matrix,
        traced_factor_labels=request.traced_factor_labels,
        reduced_matrix=partial_trace(request.matrix, request.traced_factor_labels),
    )


def decide_psd_order(request: PsdOrderRequest) -> PsdOrderResult:
    return psd_order(request.left, request.right)


__all__ = [
    "compute_kronecker_product",
    "compute_partial_trace",
    "decide_psd_order",
]
