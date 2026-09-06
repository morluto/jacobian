"""Sparse nullspace preserves source axes and exact fundamental vectors."""

import json

import pytest

from jacobian.catalog.catalog import Catalog
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices._operation_models import NullspaceResult


@pytest.mark.parametrize("free_column", [0, 99, 100])
def test_sparse_nullspace_roundtrip_and_defining_identity(free_column: int) -> None:
    n = 101
    entries = [
        {"row": j, "column": j, "value": {"num": "2", "den": "3"}}
        for j in range(n)
        if j != free_column
    ]
    matrix = {"domain": "QQ", "row_count": n, "column_count": n, "entries": entries}
    output = invoke_operation(
        "matrix.nullspace.compute", {"matrix": matrix}, Catalog.open()
    ).output
    result = NullspaceResult.model_validate_json(json.dumps(output))
    assert result.rank == n - 1 and result.nullity == 1
    assert result.free_columns == (free_column,)
    assert [value.as_fraction() for value in result.basis_vectors[0]] == [
        int(j == free_column) for j in range(n)
    ]
    assert all(
        result.basis_vectors[0][j].num == 0 for j in range(n) if j != free_column
    )


def test_sparse_nullspace_rejects_excessive_basis_before_expansion() -> None:
    with pytest.raises(OperationDomainValidationError, match="output-cell"):
        invoke_operation(
            "matrix.nullspace.compute",
            {
                "matrix": {
                    "domain": "QQ",
                    "row_count": 0,
                    "column_count": 8192,
                    "entries": [],
                }
            },
            Catalog.open(),
        )
