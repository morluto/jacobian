"""Immutable catalog declarations for subsystem-aware exact matrices."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
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
    return SubsystemPartialTraceResult._from_kernel(
        source_matrix=request.matrix,
        traced_factor_labels=request.traced_factor_labels,
        reduced_matrix=partial_trace(request.matrix, request.traced_factor_labels),
    )


def decide_psd_order(request: PsdOrderRequest) -> PsdOrderResult:
    return psd_order(request.left, request.right)


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


_QUBIT = {"label": "q", "dimension": 2}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="matrix.subsystem.kronecker_product.compute",
        title="Compute an axis-bound exact Kronecker product",
        description="Compute the exact Kronecker product of two bounded rational Hermitian "
        "matrices and concatenate their ordered subsystem factors.",
        request_type=SubsystemKroneckerProductRequest,
        result_type=SubsystemKroneckerProductResult,
        run=compute_kronecker_product,
        tags=("matrix", "kronecker-product", "subsystem-axis"),
        examples=(
            OperationExample(
                name="two_labelled_qubits",
                description="Compute the product of two diagonal 2x2 subsystem matrices; "
                "factor labels must be distinct and the product dimension at most 16.",
                input={
                    "left": {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [[_q(1), _q(0)], [_q(0), _q(2)]],
                        },
                        "factors": [_QUBIT],
                    },
                    "right": {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [[_q(3), _q(0)], [_q(0), _q(4)]],
                        },
                        "factors": [{"label": "r", "dimension": 2}],
                    },
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.subsystem.partial_trace.compute",
        title="Compute an axis-bound exact partial trace",
        description="Trace one or more named factors from a bounded rational Hermitian "
        "matrix while retaining every untraced subsystem in source order.",
        request_type=SubsystemPartialTraceRequest,
        result_type=SubsystemPartialTraceResult,
        run=compute_partial_trace,
        tags=("matrix", "partial-trace", "subsystem-axis"),
        examples=(
            OperationExample(
                name="trace_first_labelled_qubit",
                description="Trace factor q from a diagonal 4x4 two-qubit matrix; the named "
                "factor must occur once in the explicit product basis.",
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [
                                [_q(1), _q(0), _q(0), _q(0)],
                                [_q(0), _q(1), _q(0), _q(0)],
                                [_q(0), _q(0), _q(2), _q(0)],
                                [_q(0), _q(0), _q(0), _q(2)],
                            ],
                        },
                        "factors": [_QUBIT, {"label": "r", "dimension": 2}],
                    },
                    "traced_factor_labels": ["q"],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="matrix.subsystem.psd_order.decide",
        title="Decide exact PSD order on one subsystem basis",
        description="Decide whether right minus left is positive semidefinite for two "
        "bounded rational Hermitian matrices on exactly the same ordered factors.",
        request_type=PsdOrderRequest,
        result_type=PsdOrderResult,
        run=decide_psd_order,
        tags=("matrix", "positive-semidefinite", "loewner-order", "subsystem-axis"),
        examples=(
            OperationExample(
                name="diagonal_psd_order",
                description="Decide 0 <= diag(1, 2) on one labelled subsystem; both matrices "
                "must carry exactly equal factors and basis linearization.",
                input={
                    "left": {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [[_q(0), _q(0)], [_q(0), _q(0)]],
                        },
                        "factors": [_QUBIT],
                    },
                    "right": {
                        "matrix": {
                            "domain": "QQ",
                            "entries": [[_q(1), _q(0)], [_q(0), _q(2)]],
                        },
                        "factors": [_QUBIT],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
