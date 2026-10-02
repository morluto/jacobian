"""Exact public centralizer bases span the identity in every kernel regime."""

import json

import pytest
from sympy import Matrix, eye

from jacobian.catalog.catalog import Catalog
from jacobian.dispatch import invoke_operation
from jacobian.math.matrices.canonical_forms._models import CentralizerResult


@pytest.mark.parametrize(
    ("rows", "dimension", "identity_member"),
    [
        (((2, 0), (0, 2)), 4, False),
        (((0, 0, 0), (0, 1, 0), (0, 0, 2)), 3, False),
        (((2, 1), (0, 2)), 2, True),
        (((0, 1, 0), (0, 0, 1), (0, 0, 0)), 3, True),
    ],
)
def test_public_centralizer_basis_spans_identity(
    rows: tuple[tuple[int, ...], ...], dimension: int, identity_member: bool
) -> None:
    result = invoke_operation(
        "matrix.centralizer.compute",
        {
            "matrix": {
                "entries": [
                    [{"num": str(value), "den": "1"} for value in row] for row in rows
                ]
            }
        },
        Catalog.open(),
    )
    decoded = CentralizerResult.model_validate_json(json.dumps(result.output))
    source = Matrix(rows)
    order = len(rows)
    basis = [
        Matrix([[entry.as_fraction() for entry in row] for row in matrix.entries])
        for matrix in decoded.basis
    ]
    assert decoded.dimension == len(basis) == dimension
    assert all(source * matrix == matrix * source for matrix in basis)
    vectors = Matrix.hstack(*(matrix.reshape(order * order, 1) for matrix in basis))
    assert vectors.rank() == dimension
    assert vectors.row_join(eye(order).reshape(order * order, 1)).rank() == dimension
    assert (eye(order) in basis) is identity_member
