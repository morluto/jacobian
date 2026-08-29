"""Dispatch boundaries for exact rational matrix operations."""

from __future__ import annotations

from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.matrices._operation_models import (
    MAX_CHARACTERISTIC_POLYNOMIAL_ORDER,
    MAX_DETERMINANT_MATRIX_DIMENSION,
    MAX_PERMANENT_MATRIX_ORDER,
)
from jacobian.math.matrices.values import (
    MAX_EXACT_LINEAR_MATRIX_AXIS,
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
)


def _identity_payload(order: int) -> dict[str, Any]:
    return {
        "matrix": {
            "entries": [
                [
                    {"num": "1" if row == column else "0", "den": "1"}
                    for column in range(order)
                ]
                for row in range(order)
            ]
        }
    }


def _partial_trace_payload(
    traced_dimension: int, kept_dimension: int
) -> dict[str, Any]:
    total = traced_dimension * kept_dimension
    payload = _identity_payload(total)
    payload["traced_dimension"] = traced_dimension
    payload["kept_dimension"] = kept_dimension
    return payload


def _oversized_partial_trace_payload() -> dict[str, Any]:
    """Smallest composite exceeding the computation dimension within the wire order."""
    for total in range(MAX_MATRIX_DIMENSION + 1, MAX_RATIONAL_MATRIX_ORDER + 1):
        for kept_dimension in range(1, total + 1):
            if total % kept_dimension:
                continue
            traced_dimension = total // kept_dimension
            if max(traced_dimension, kept_dimension) <= MAX_MATRIX_DIMENSION:
                return _partial_trace_payload(
                    traced_dimension=traced_dimension,
                    kept_dimension=kept_dimension,
                )
    raise AssertionError(
        "no composite shape fits the wire order budget while exceeding "
        "the computation dimension"
    )


@pytest.mark.parametrize(
    ("operation_id", "payload", "maximum_order", "error_type"),
    (
        (
            "matrix.characteristic_polynomial.compute",
            _identity_payload(MAX_CHARACTERISTIC_POLYNOMIAL_ORDER + 1),
            MAX_CHARACTERISTIC_POLYNOMIAL_ORDER,
            OperationRequestValidationError,
        ),
        (
            "matrix.permanent.compute",
            _identity_payload(MAX_MATRIX_DIMENSION + 1),
            MAX_PERMANENT_MATRIX_ORDER,
            OperationRequestValidationError,
        ),
        (
            "matrix.rank.compute",
            _identity_payload(MAX_EXACT_LINEAR_MATRIX_AXIS + 1),
            MAX_EXACT_LINEAR_MATRIX_AXIS,
            OperationDomainValidationError,
        ),
        (
            "matrix.normal_form.rref.compute",
            _identity_payload(MAX_EXACT_LINEAR_MATRIX_AXIS + 1),
            MAX_EXACT_LINEAR_MATRIX_AXIS,
            OperationDomainValidationError,
        ),
        (
            "matrix.nullspace.compute",
            _identity_payload(MAX_EXACT_LINEAR_MATRIX_AXIS + 1),
            MAX_EXACT_LINEAR_MATRIX_AXIS,
            OperationDomainValidationError,
        ),
        (
            "matrix.partial_trace.compute",
            _oversized_partial_trace_payload(),
            MAX_MATRIX_DIMENSION,
            OperationRequestValidationError,
        ),
    ),
)
def test_dispatch_rejects_requests_above_the_computation_dimension(
    operation_id: str,
    payload: dict[str, Any],
    maximum_order: int,
    error_type: type[ValueError],
) -> None:
    with pytest.raises(error_type) as excinfo:
        invoke_operation(operation_id, payload, Catalog.open())
    message = (
        str(excinfo.value.cause)
        if isinstance(excinfo.value, OperationRequestValidationError)
        else str(excinfo.value)
    )
    assert f"limited to {maximum_order} rows and columns" in message


def test_dispatch_returns_typed_results_at_the_boundary_order() -> None:
    result = invoke_operation(
        "matrix.characteristic_polynomial.compute",
        _identity_payload(MAX_MATRIX_DIMENSION),
        Catalog.open(),
    )
    assert result.output["degree"] == MAX_MATRIX_DIMENSION
    assert len(result.output["coefficients_descending"]) == MAX_MATRIX_DIMENSION + 1


def test_native_and_dispatch_determinants_share_the_canonical_boundary() -> None:
    import sympy

    from jacobian.math import matrices

    native = matrices.determinant(sympy.eye(MAX_DETERMINANT_MATRIX_DIMENSION))
    dispatched = invoke_operation(
        "matrix.determinant.compute",
        _identity_payload(MAX_DETERMINANT_MATRIX_DIMENSION),
        Catalog.open(),
    )

    assert dispatched.output["determinant"] == {"num": str(native), "den": "1"}


def test_dispatch_preserves_the_typed_singular_inverse_rejection() -> None:
    """A proved singularity is an owner error, never an SDK/host fault."""

    with pytest.raises(OperationDomainValidationError) as excinfo:
        invoke_operation(
            "matrix.inverse.compute",
            {"matrix": {"entries": [["1", "2"], ["2", "4"]]}},
            Catalog.open(),
        )

    assert excinfo.value.errors() == (
        {
            "loc": ("matrix",),
            "type": "matrix.singular_matrix",
            "msg": "matrix is singular; inverse does not exist",
        },
    )
