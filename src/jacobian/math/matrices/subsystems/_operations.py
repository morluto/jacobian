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
    return SubsystemPartialTraceResult._from_kernel(
        source_matrix=request.matrix,
        traced_factor_labels=request.traced_factor_labels,
        reduced_matrix=_partial_trace_kernel(
            request.matrix,
            request.traced_factor_labels,
        ),
    )


def decide_psd_order(request: PsdOrderRequest) -> PsdOrderResult:
    return _psd_order_kernel(request.left, request.right)


def _verify_partial_trace_result(result: SubsystemPartialTraceResult) -> bool:
    """Verify one independently supplied trace claim in the admitted envelope."""

    try:
        request = SubsystemPartialTraceRequest(
            matrix=result.source_matrix,
            traced_factor_labels=result.traced_factor_labels,
        )
    except ValueError:
        return False
    return (
        _partial_trace_kernel(request.matrix, request.traced_factor_labels)
        == result.reduced_matrix
    )


def _verify_psd_order_result(result: PsdOrderResult) -> bool:
    """Verify one independently supplied PSD-order claim in the admitted envelope."""

    try:
        request = PsdOrderRequest(left=result.left, right=result.right)
    except ValueError:
        return False
    expected = _psd_order_kernel(request.left, request.right)
    return (
        result.difference,
        result.inertia,
        result.is_less_or_equal,
        result.negative_witness,
    ) == (
        expected.difference,
        expected.inertia,
        expected.is_less_or_equal,
        expected.negative_witness,
    )


__all__ = [
    "compute_kronecker_product",
    "compute_partial_trace",
    "decide_psd_order",
]
