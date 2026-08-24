"""Immutable catalog declarations for subsystem-aware exact matrices."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.matrices.subsystems._models import (
    PsdOrderRequest,
    PsdOrderResult,
    SubsystemKroneckerProductRequest,
    SubsystemKroneckerProductResult,
    SubsystemPartialTraceRequest,
    SubsystemPartialTraceResult,
)
from jacobian.math.matrices.subsystems._operations import (
    compute_kronecker_product,
    compute_partial_trace,
    decide_psd_order,
)


def _operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...],
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


def _q(value: int) -> dict[str, str]:
    return {"num": str(value), "den": "1"}


_QUBIT = {"label": "q", "dimension": 2}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _operation(
        "matrix.subsystem.kronecker_product.compute",
        "Compute an axis-bound exact Kronecker product",
        "Compute the exact Kronecker product of two bounded rational Hermitian "
        "matrices and concatenate their ordered subsystem factors.",
        SubsystemKroneckerProductRequest,
        SubsystemKroneckerProductResult,
        compute_kronecker_product,
        "matrix",
        "kronecker-product",
        "subsystem-axis",
        examples=(
            example(
                "two_labelled_qubits",
                "Compute the product of two diagonal 2x2 subsystem matrices; "
                "factor labels must be distinct and the product dimension at most 16.",
                {
                    "left": {
                        "matrix": {"entries": [[_q(1), _q(0)], [_q(0), _q(2)]]},
                        "factors": [_QUBIT],
                    },
                    "right": {
                        "matrix": {"entries": [[_q(3), _q(0)], [_q(0), _q(4)]]},
                        "factors": [{"label": "r", "dimension": 2}],
                    },
                },
            ),
        ),
    ),
    _operation(
        "matrix.subsystem.partial_trace.compute",
        "Compute an axis-bound exact partial trace",
        "Trace one or more named factors from a bounded rational Hermitian "
        "matrix while retaining every untraced subsystem in source order.",
        SubsystemPartialTraceRequest,
        SubsystemPartialTraceResult,
        compute_partial_trace,
        "matrix",
        "partial-trace",
        "subsystem-axis",
        examples=(
            example(
                "trace_first_labelled_qubit",
                "Trace factor q from a diagonal 4x4 two-qubit matrix; the named "
                "factor must occur once in the explicit product basis.",
                {
                    "matrix": {
                        "matrix": {
                            "entries": [
                                [_q(1), _q(0), _q(0), _q(0)],
                                [_q(0), _q(1), _q(0), _q(0)],
                                [_q(0), _q(0), _q(2), _q(0)],
                                [_q(0), _q(0), _q(0), _q(2)],
                            ]
                        },
                        "factors": [_QUBIT, {"label": "r", "dimension": 2}],
                    },
                    "traced_factor_labels": ["q"],
                },
            ),
        ),
    ),
    _operation(
        "matrix.subsystem.psd_order.decide",
        "Decide exact PSD order on one subsystem basis",
        "Decide whether right minus left is positive semidefinite for two "
        "bounded rational Hermitian matrices on exactly the same ordered factors.",
        PsdOrderRequest,
        PsdOrderResult,
        decide_psd_order,
        "matrix",
        "positive-semidefinite",
        "loewner-order",
        "subsystem-axis",
        examples=(
            example(
                "diagonal_psd_order",
                "Decide 0 <= diag(1, 2) on one labelled subsystem; both matrices "
                "must carry exactly equal factors and basis linearization.",
                {
                    "left": {
                        "matrix": {"entries": [[_q(0), _q(0)], [_q(0), _q(0)]]},
                        "factors": [_QUBIT],
                    },
                    "right": {
                        "matrix": {"entries": [[_q(1), _q(0)], [_q(0), _q(2)]]},
                        "factors": [_QUBIT],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
