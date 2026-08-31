"""Tests for finite-field matrix rank."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from jacobian.math.finite_fields._matrix_rank import compute_rank
from jacobian.math.finite_fields._matrix_rank_models import MatrixRankRequest
from jacobian.math.finite_fields.operations import matrix_rank
from jacobian.math.finite_fields.values import (
    Axis,
    AxisBoundMatrix,
    FiniteFieldElement,
    FiniteFieldPresentation,
)


def _f2() -> FiniteFieldPresentation:
    return FiniteFieldPresentation(
        characteristic=2, modulus_coefficients=(0, 1), generator="a"
    )


def _f4() -> FiniteFieldPresentation:
    return FiniteFieldPresentation(
        characteristic=2, modulus_coefficients=(1, 1, 1), generator="a"
    )


def _matrix(
    fp: FiniteFieldPresentation,
    rows: list[list[tuple[int, ...]]],
    row_labels: list[str],
    col_labels: list[str],
) -> AxisBoundMatrix:
    return AxisBoundMatrix(
        presentation=fp,
        row_axis=Axis(name="rows", labels=tuple(row_labels)),
        column_axis=Axis(name="cols", labels=tuple(col_labels)),
        entries=tuple(
            tuple(
                FiniteFieldElement(presentation=fp, coordinates=tuple(coords))
                for coords in row
            )
            for row in rows
        ),
    )


def test_rank_one_over_f2() -> None:
    fp = _f2()
    m = _matrix(fp, [[[1], [1]], [[1], [1]]], ["r0", "r1"], ["c0", "c1"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 1


def test_identity_rank_two_over_f2() -> None:
    fp = _f2()
    m = _matrix(fp, [[[1], [0]], [[0], [1]]], ["r0", "r1"], ["c0", "c1"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 2


def test_zero_matrix_rank_zero() -> None:
    fp = _f2()
    m = _matrix(fp, [[[0], [0]], [[0], [0]]], ["r0", "r1"], ["c0", "c1"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 0


def test_rectangular_rank_deficient() -> None:
    """A 2x3 matrix with two independent rows has rank 2."""
    fp = _f2()
    m = _matrix(
        fp,
        [[[1], [0], [1]], [[0], [1], [1]]],
        ["r0", "r1"],
        ["c0", "c1", "c2"],
    )
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 2


def test_extension_field_rank_one() -> None:
    """A 1x1 matrix with nonzero GF(4) element has rank 1."""
    fp = _f4()
    m = _matrix(fp, [[[0, 1]]], ["r0"], ["c0"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 1


def test_extension_field_zero_rank() -> None:
    """A 1x1 zero matrix over GF(4) has rank 0."""
    fp = _f4()
    m = _matrix(fp, [[[0, 0]]], ["r0"], ["c0"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 0


def test_extension_field_identity() -> None:
    """Identity matrix over GF(4) has full rank."""
    fp = _f4()
    m = _matrix(
        fp,
        [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
        ["r0", "r1"],
        ["c0", "c1"],
    )
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.rank == 2


def test_pivot_labels_preserved() -> None:
    """Pivot row and column labels are from the source axes."""
    fp = _f2()
    m = _matrix(fp, [[[1], [1]], [[1], [1]]], ["r0", "r1"], ["c0", "c1"])
    result = compute_rank(MatrixRankRequest(matrix=m))
    assert result.pivot_rows == ("r0",)
    assert result.pivot_columns == ("c0",)


def test_pivot_labels_follow_row_swaps() -> None:
    fp = _f2()
    m = _matrix(fp, [[[0]], [[1]]], ["r0", "r1"], ["c0"])
    result = matrix_rank(m)
    assert result.rank == 1
    assert result.pivot_rows == ("r1",)


def test_pivot_columns_follow_source_axis_order() -> None:
    fp = _f2()
    m = _matrix(fp, [[[1], [0]], [[0], [1]]], ["r0", "r1"], ["c0", "c1"])
    with pytest.raises(ValidationError, match="pivot columns must follow"):
        type(matrix_rank(m))(
            matrix=m,
            rank=2,
            pivot_rows=("r0", "r1"),
            pivot_columns=("c1", "c0"),
        )


def test_rank_invariance_under_row_ops() -> None:
    """Rank is invariant under adding one row to another (over F_2)."""
    fp = _f2()
    m1 = _matrix(fp, [[[1], [0]], [[0], [1]]], ["r0", "r1"], ["c0", "c1"])
    m2 = _matrix(fp, [[[1], [0]], [[1], [1]]], ["r0", "r1"], ["c0", "c1"])
    assert compute_rank(MatrixRankRequest(matrix=m1)).rank == 2
    assert compute_rank(MatrixRankRequest(matrix=m2)).rank == 2
