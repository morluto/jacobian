"""Dispatch boundaries for exact rational matrix operations."""

from __future__ import annotations

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation


def _identity_payload(order: int) -> dict:
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


def _partial_trace_payload(traced_dimension: int, kept_dimension: int) -> dict:
    total = traced_dimension * kept_dimension
    payload = _identity_payload(total)
    payload["traced_dimension"] = traced_dimension
    payload["kept_dimension"] = kept_dimension
    return payload


@pytest.mark.parametrize(
    ("operation_id", "payload"),
    (
        (
            "matrix.characteristic_polynomial.compute",
            _identity_payload(33),
        ),
        ("matrix.permanent.compute", _identity_payload(33)),
        ("matrix.rank.compute", _identity_payload(33)),
        ("matrix.normal_form.rref.compute", _identity_payload(33)),
        ("matrix.nullspace.compute", _identity_payload(33)),
        (
            "matrix.partial_trace.compute",
            _partial_trace_payload(traced_dimension=11, kept_dimension=3),
        ),
    ),
)
def test_dispatch_rejects_requests_above_the_computation_dimension(
    operation_id: str, payload: dict
) -> None:
    with pytest.raises(OperationRequestValidationError) as excinfo:
        invoke_operation(operation_id, payload, Catalog.open())
    assert "limited to 32 rows and columns" in str(excinfo.value.cause)


def test_dispatch_returns_typed_results_at_the_boundary_order() -> None:
    result = invoke_operation(
        "matrix.characteristic_polynomial.compute",
        _identity_payload(32),
        Catalog.open(),
    )
    assert result.output["degree"] == 32
    assert len(result.output["coefficients_descending"]) == 33
