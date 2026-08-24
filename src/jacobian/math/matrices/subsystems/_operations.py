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
    _kronecker_product_kernel,
    _partial_trace_kernel,
    _psd_order_kernel,
)


def compute_kronecker_product(
    request: SubsystemKroneckerProductRequest,
) -> SubsystemKroneckerProductResult:
    return SubsystemKroneckerProductResult(
        product=_kronecker_product_kernel(request.left, request.right)
    )


def compute_partial_trace(
    request: SubsystemPartialTraceRequest,
) -> SubsystemPartialTraceResult:
    return SubsystemPartialTraceResult(
        source_matrix=request.matrix,
        traced_factor_labels=request.traced_factor_labels,
        reduced_matrix=_partial_trace_kernel(
            request.matrix,
            request.traced_factor_labels,
        ),
    )


def decide_psd_order(request: PsdOrderRequest) -> PsdOrderResult:
    return _psd_order_kernel(request.left, request.right)


__all__ = [
    "compute_kronecker_product",
    "compute_partial_trace",
    "decide_psd_order",
]
