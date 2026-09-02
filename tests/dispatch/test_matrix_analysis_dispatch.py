"""Dispatch boundaries for exact matrix inertia analysis."""

from __future__ import annotations

import pytest

from jacobian.canonical import canonicalize_json
from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION


def _high_digit_dense_inertia_payload() -> dict[str, object]:
    value = {"num": "9" * 4096, "den": "1"}
    return {
        "matrix": {
            "domain": "QQ",
            "entries": [[value for _ in range(16)] for _ in range(16)],
        }
    }


def test_dispatch_rejects_inertia_above_digit_work_bound() -> None:
    with pytest.raises(OperationDomainValidationError) as excinfo:
        invoke_operation(
            "matrix.inertia.compute",
            _high_digit_dense_inertia_payload(),
            Catalog.open(),
        )
    assert "digit-work bound" in str(excinfo.value)


def test_dispatch_admits_order_33_diagonal_inertia_request() -> None:
    # Order 33 exceeds the shared computation dimension but stays inside the
    # canonical dense rational-matrix order envelope, so a small-entry source
    # there must be admitted end to end with a typed source-bound result.
    zero = {"num": "0", "den": "1"}
    one = {"num": "1", "den": "1"}
    payload = {
        "matrix": {
            "domain": "QQ",
            "entries": [
                [one if row == column else zero for column in range(33)]
                for row in range(33)
            ],
        }
    }
    result = invoke_operation("matrix.inertia.compute", payload, Catalog.open())

    assert result.output["n_positive"] == 33
    assert result.output["n_negative"] == 0
    assert result.output["n_zero"] == 0
    assert result.output["definiteness"] == "positive_definite"
    assert len(result.output["matrix"]["entries"]) == 33


def test_large_fitting_inertia_request_returns_typed_result() -> None:
    digits = "9" * 4096
    zero = {"num": "0", "den": "1"}
    large = {"num": digits, "den": "1"}
    payload = {
        "matrix": {
            "domain": "QQ",
            "entries": [
                [
                    large if row == column else zero
                    for column in range(MAX_MATRIX_DIMENSION)
                ]
                for row in range(MAX_MATRIX_DIMENSION)
            ],
        }
    }
    assert len(canonicalize_json(payload)) > 100_000
    result = invoke_operation("matrix.inertia.compute", payload, Catalog.open())
    assert result.output["n_positive"] == MAX_MATRIX_DIMENSION
    assert result.output["n_negative"] == 0
    assert result.output["n_zero"] == 0
    assert result.output["definiteness"] == "positive_definite"
    matrix = result.output["matrix"]
    assert matrix["domain"] == "QQ"
    assert len(matrix["entries"]) == MAX_MATRIX_DIMENSION
