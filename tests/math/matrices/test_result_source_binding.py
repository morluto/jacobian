"""Regression tests binding rational RREF, rank, and nullspace to their source."""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from jacobian._exact import CanonicalRational
from jacobian.math.matrices._operation_models import (
    MAX_DETERMINANT_MATRIX_DIMENSION,
    MatrixDeterminantRequest,
    MatrixRankRequest,
    MatrixRankResult,
    NullspaceResult,
    RationalMatrixProductRequest,
    RationalMatrixRequest,
    RrefResult,
)
from jacobian.math.matrices._tools import (
    compute_determinant,
    compute_nullspace,
    compute_product,
    compute_rank,
    compute_rref,
)
from jacobian.math.matrices.values import (
    MAX_MATRIX_DIMENSION,
    MAX_RATIONAL_MATRIX_ORDER,
    RationalMatrix,
    rational_matrix_from_fractions,
)


def _matrix(rows: list[list[str]]) -> RationalMatrix:
    return RationalMatrix.model_validate(
        {
            "domain": "QQ",
            "entries": [
                [
                    {
                        "num": value.split("/")[0],
                        "den": value.split("/")[1] if "/" in value else "1",
                    }
                    for value in row
                ]
                for row in rows
            ],
        }
    )


def test_rational_matrix_from_fractions_preserves_canonical_exact_entries() -> None:
    matrix = rational_matrix_from_fractions(
        ((Fraction(-6, 8), Fraction(5)), (Fraction(0), Fraction(7, 12)))
    )

    assert matrix.model_dump(mode="json") == {
        "domain": "QQ",
        "entries": [
            [{"num": "-3", "den": "4"}, {"num": "5", "den": "1"}],
            [{"num": "0", "den": "1"}, {"num": "7", "den": "12"}],
        ],
    }


def test_producer_results_replay_across_shapes() -> None:
    """Zero, rectangular, rank-deficient, and full-rank sources stay bound."""

    shapes = (
        _matrix([["0", "0"], ["0", "0"]]),
        _matrix([["1", "2", "3"], ["4", "5", "6"]]),
        _matrix([["1", "2"], ["2", "4"], ["3", "6"]]),
        _matrix([["1/2", "0"], ["0", "1/3"]]),
    )
    for matrix in shapes:
        request = RationalMatrixRequest(matrix=matrix)
        rref = compute_rref(request)
        assert rref.matrix == matrix

        rank_request = MatrixRankRequest(matrix=matrix)
        rank = compute_rank(rank_request)
        assert rank.matrix == matrix
        assert rank.rank == len(rank.pivot_columns)

        nullspace = compute_nullspace(request)
        assert nullspace.matrix == matrix
        assert nullspace.rank + nullspace.nullity == nullspace.ambient_dimension
        assert len(nullspace.basis_vectors) == nullspace.nullity


@pytest.mark.parametrize(
    "rows",
    (
        [["1", "2", "3"], ["4", "5", "6"]],
        [["0", "0", "0"], ["1", "0", "0"]],
        [["1/2", "1/3"], ["1/5", "1/7"], ["1", "1"]],
    ),
)
def test_serialized_results_round_trip(rows: list[list[str]]) -> None:
    matrix = _matrix(rows)
    rref = compute_rref(RationalMatrixRequest(matrix=matrix))
    assert RrefResult.model_validate(rref.model_dump()) == rref

    rank = compute_rank(MatrixRankRequest(matrix=matrix))
    assert MatrixRankResult.model_validate(rank.model_dump()) == rank

    nullspace = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    assert NullspaceResult.model_validate(nullspace.model_dump()) == nullspace


def test_producer_to_serialized_interoperability() -> None:
    """Serialized RREF agrees with independently produced rank and nullspace."""

    matrix = _matrix([["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"]])
    rref = RrefResult.model_validate(
        compute_rref(RationalMatrixRequest(matrix=matrix)).model_dump()
    )
    rank = compute_rank(MatrixRankRequest(matrix=matrix))
    nullspace = compute_nullspace(RationalMatrixRequest(matrix=matrix))
    assert rref.rank == rank.rank == nullspace.rank == 2
    assert list(rref.pivot_columns) == list(rank.pivot_columns)
    assert list(rref.free_columns) == list(nullspace.free_columns)


def test_product_value_feeds_determinant_without_reencoding() -> None:
    """A matrix producer's canonical value is the determinant input value."""

    left = _matrix([["1", "2"], ["3", "4"]])
    right = _matrix([["0", "1"], ["1", "0"]])
    product = compute_product(RationalMatrixProductRequest(left=left, right=right))

    # Transport composition parses the producer's canonical payload directly,
    # and native composition retains its exact owner type unchanged.
    transported = MatrixDeterminantRequest.model_validate(
        {"matrix": product.product.model_dump(mode="json")}
    )
    assert transported.matrix == product.product

    request = MatrixDeterminantRequest(matrix=product.product)
    assert request.matrix is product.product
    assert compute_determinant(request).determinant == CanonicalRational(
        num="2", den="1"
    )


def _identity_entries(size: int) -> tuple[tuple[CanonicalRational, ...], ...]:
    one = CanonicalRational(num="1", den="1")
    zero = CanonicalRational(num="0", den="1")
    return tuple(
        tuple(one if index == column else zero for column in range(size))
        for index in range(size)
    )


def test_request_admission_rejects_matrices_above_the_computation_dimension() -> None:
    oversized = RationalMatrix(entries=_identity_entries(MAX_MATRIX_DIMENSION + 1))
    with pytest.raises(ValidationError):
        RationalMatrixRequest(matrix=oversized)
    with pytest.raises(ValidationError):
        MatrixRankRequest(matrix=oversized)


def test_request_admission_rejects_one_oversized_axis() -> None:
    tall = RationalMatrix(
        entries=tuple(
            tuple(
                CanonicalRational(num=str(column + 1), den="1") for column in range(2)
            )
            for _ in range(MAX_MATRIX_DIMENSION + 1)
        )
    )
    with pytest.raises(ValidationError):
        RationalMatrixRequest(matrix=tall)
    with pytest.raises(ValidationError):
        MatrixRankRequest(matrix=tall)


def test_request_admission_keeps_the_boundary_computation_dimension() -> None:
    boundary = RationalMatrix(entries=_identity_entries(MAX_MATRIX_DIMENSION))
    request = MatrixRankRequest(matrix=boundary)
    rank = compute_rank(request)
    assert rank.rank == MAX_MATRIX_DIMENSION
    assert len(rank.pivot_columns) == MAX_MATRIX_DIMENSION
    assert rank.matrix == boundary


def test_determinant_accepts_the_canonical_matrix_boundary() -> None:
    assert MAX_RATIONAL_MATRIX_ORDER == MAX_DETERMINANT_MATRIX_DIMENSION
    matrix = RationalMatrix(entries=_identity_entries(MAX_RATIONAL_MATRIX_ORDER))
    assert MatrixDeterminantRequest(matrix=matrix).matrix is matrix


def test_raw_preflight_keeps_32_and_64_axes_and_rejects_the_next_axis() -> None:
    def wire_identity(order: int) -> list[list[dict[str, str]]]:
        return [
            [{"num": str(int(row == column)), "den": "1"} for column in range(order)]
            for row in range(order)
        ]

    rank_boundary = {"matrix": {"entries": wire_identity(MAX_MATRIX_DIMENSION)}}
    assert MatrixRankRequest.model_validate(
        rank_boundary
    ).matrix.entries == _identity_entries(MAX_MATRIX_DIMENSION)
    with pytest.raises(ValidationError):
        MatrixRankRequest.model_validate(
            {"matrix": {"entries": wire_identity(MAX_MATRIX_DIMENSION + 1)}}
        )

    determinant_boundary = {
        "matrix": {"entries": wire_identity(MAX_DETERMINANT_MATRIX_DIMENSION)}
    }
    assert MatrixDeterminantRequest.model_validate(
        determinant_boundary
    ).matrix.entries == (_identity_entries(MAX_DETERMINANT_MATRIX_DIMENSION))
    with pytest.raises(ValidationError):
        MatrixDeterminantRequest.model_validate(
            {"matrix": {"entries": wire_identity(MAX_DETERMINANT_MATRIX_DIMENSION + 1)}}
        )


def test_raw_preflight_rejects_a_257_digit_operation_scalar() -> None:
    with pytest.raises(ValidationError):
        MatrixRankRequest.model_validate(
            {
                "matrix": {
                    "entries": [
                        [{"num": "9" * 257, "den": "1"}],
                    ]
                }
            }
        )
