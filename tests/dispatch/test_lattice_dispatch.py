"""Dispatch boundaries for integer-lattice operations."""

from typing import Any

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import OperationRequestValidationError, invoke_operation
from jacobian.math.matrices.values import MAX_MATRIX_DIMENSION


def _identity_payload(order: int) -> dict[str, Any]:
    return {
        "basis": {
            "entries": [
                ["1" if row == column else "0" for column in range(order)]
                for row in range(order)
            ]
        }
    }


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
