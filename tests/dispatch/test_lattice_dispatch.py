"""Dispatch boundaries for integer-lattice operations."""

from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.lattices._hnf import compute_hermite_normal_form
from jacobian.math.lattices._lattice import reduce_lattice_basis
from jacobian.math.lattices._models import (
    HermiteNormalFormRequest,
    LatticeReductionRequest,
)
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION, IntegerMatrix


def _identity_payload(order: int) -> dict[str, Any]:
    return {
        "basis": {
            "entries": [
                ["1" if row == column else "0" for column in range(order)]
                for row in range(order)
            ]
        }
    }


def _identity_entries(order: int) -> list[list[str]]:
    return [
        ["1" if row == column else "0" for column in range(order)]
        for row in range(order)
    ]


def test_lattice_reduction_admits_before_lll_backend() -> None:
    basis = IntegerMatrix.model_validate(
        {"entries": _identity_entries(MAX_MATRIX_DIMENSION + 1)}
    )
    request = LatticeReductionRequest.model_construct(basis=basis)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        reduce_lattice_basis(request)
    error = exc_info.value.errors()[0]
    assert error["type"] == "lattice.budget_exceeded"
    assert error["loc"] == ("basis",)
    assert str(MAX_MATRIX_DIMENSION) in error["msg"]


def test_hermite_admits_before_hnf_backend() -> None:
    matrix = IntegerMatrix.model_validate(
        {"entries": _identity_entries(MAX_MATRIX_DIMENSION + 1)}
    )
    request = HermiteNormalFormRequest.model_construct(matrix=matrix)
    with pytest.raises(OperationDomainValidationError) as exc_info:
        compute_hermite_normal_form(request)
    error = exc_info.value.errors()[0]
    assert error["type"] == "lattice.budget_exceeded"
    assert error["loc"] == ("matrix",)


def test_dispatch_rejects_lll_above_the_lattice_axis() -> None:
    with pytest.raises(OperationRequestValidationError) as exc_info:
        invoke_operation(
            "lattice.basis.reduce",
            _identity_payload(MAX_MATRIX_DIMENSION + 1),
            Catalog.open(),
        )
    assert f"limited to {MAX_MATRIX_DIMENSION} rows and columns" in str(
        exc_info.value.cause
    )
