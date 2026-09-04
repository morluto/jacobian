"""Exact rank contracts for canonical coordinate-sparse rational matrices."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import islice
from math import isqrt
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sympy import primerange
from tests.fixtures.accounting import assert_charged_work_parity

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.matrices import operations
from jacobian.math.matrices._operation_models import (
    MAX_EXACT_LINEAR_MATRIX_WORK,
    MAX_FLINT_SPARSE_RANK_WORK,
    MAX_INPUT_SCALAR_DIGITS,
    MAX_SPARSE_RANK_INTERMEDIATE_CELLS,
    MatrixRankRequest,
    MatrixRankResult,
)
from jacobian.math.matrices._tools import compute_rank
from jacobian.math.matrices.operations import rank_result
from jacobian.math.matrices.values import (
    MAX_SPARSE_RATIONAL_MATRIX_AXIS,
    RationalMatrix,
    SparseRationalMatrix,
    SparseRationalMatrixEntry,
    rational_matrix_from_fractions,
    sparse_rational_matrix_from_dense,
)


def _entry(
    row: int,
    column: int,
    value: Fraction = Fraction(1),
) -> SparseRationalMatrixEntry:
    return SparseRationalMatrixEntry(
        row=row,
        column=column,
        value=CanonicalRational.from_fraction(value),
    )


def _connected_fill_matrix(order: int) -> SparseRationalMatrix:
    entries = [_entry(0, column) for column in range(order)]
    for row in range(1, order):
        entries.extend((_entry(row, 0), _entry(row, row, Fraction(2))))
    return SparseRationalMatrix(
        row_count=order,
        column_count=order,
        entries=tuple(entries),
    )


def _connected_fill_work(order: int) -> int:
    return 2 * order + (3 * order - 2) + order**3


@pytest.mark.parametrize(
    ("row_count", "column_count"),
    ((0, 128), (128, 0), (0, 0)),
)
def test_sparse_rank_retains_empty_axes_from_strict_json(
    row_count: int, column_count: int
) -> None:
    payload = {
        "matrix": {
            "domain": "QQ",
            "row_count": row_count,
            "column_count": column_count,
            "entries": [],
        }
    }
    request = MatrixRankRequest.model_validate_json(json.dumps(payload), strict=True)

    result = compute_rank(request)

    assert result.matrix is request.matrix
    assert result.rank == 0
    assert result.pivot_columns == ()
    assert result.matrix.model_dump(mode="json") == payload["matrix"]
    assert (
        MatrixRankResult.model_validate_json(result.model_dump_json(), strict=True)
        == result
    )


def test_sparse_rank_reports_a_pivot_at_column_127() -> None:
    matrix = SparseRationalMatrix(
        row_count=1,
        column_count=128,
        entries=(_entry(0, 127),),
    )

    result = rank_result(matrix)

    assert result.matrix is matrix
    assert result.rank == 1
    assert result.pivot_columns == (127,)


def test_sparse_rank_pivots_define_exact_row_reduction() -> None:
    matrix = SparseRationalMatrix(
        row_count=3,
        column_count=128,
        entries=(
            _entry(0, 1),
            _entry(0, 127, Fraction(2)),
            _entry(1, 1, Fraction(2)),
            _entry(1, 127, Fraction(4)),
            _entry(2, 5, Fraction(3)),
            _entry(2, 127),
        ),
    )

    result = rank_result(matrix)

    assert result.rank == 2
    assert result.pivot_columns == (1, 5)


def test_dense_and_sparse_rank_paths_agree_without_changing_representation() -> None:
    dense = rational_matrix_from_fractions(
        (
            (Fraction(1, 2), Fraction(0), Fraction(2), Fraction(1)),
            (Fraction(1), Fraction(0), Fraction(4), Fraction(2)),
            (Fraction(0), Fraction(3), Fraction(1), Fraction(0)),
        )
    )
    sparse = sparse_rational_matrix_from_dense(dense)

    dense_result = rank_result(dense)
    sparse_result = rank_result(sparse)

    assert isinstance(dense_result.matrix, RationalMatrix)
    assert dense_result.matrix is dense
    assert isinstance(sparse_result.matrix, SparseRationalMatrix)
    assert sparse_result.matrix is sparse
    assert sparse_result.rank == dense_result.rank
    assert sparse_result.pivot_columns == dense_result.pivot_columns


def test_sparse_rank_admits_a_2016_by_128_active_support_profile() -> None:
    row_count = 2_016
    column_count = 128
    independent_rows = 125
    entries: list[SparseRationalMatrixEntry] = []
    for row in range(row_count):
        basis_column = row % independent_rows
        entries.append(_entry(row, basis_column))
        if basis_column < column_count - independent_rows:
            entries.append(_entry(row, independent_rows + basis_column))
    matrix = SparseRationalMatrix(
        row_count=row_count,
        column_count=column_count,
        entries=tuple(entries),
    )

    result = rank_result(matrix)

    assert result.rank == independent_rows
    assert result.pivot_columns == tuple(range(independent_rows))
    assert result.matrix is matrix


def test_sparse_rank_request_preflights_its_scalar_budget() -> None:
    with pytest.raises(
        ValidationError, match=rf"{MAX_INPUT_SCALAR_DIGITS} decimal digits"
    ):
        MatrixRankRequest.model_validate(
            {
                "matrix": {
                    "row_count": 1,
                    "column_count": 1,
                    "entries": [
                        {
                            "row": 0,
                            "column": 0,
                            "value": {
                                "num": "9" * (MAX_INPUT_SCALAR_DIGITS + 1),
                                "den": "1",
                            },
                        }
                    ],
                }
            }
        )


def test_sparse_rank_accepts_a_max_axis_diagonal() -> None:
    order = MAX_SPARSE_RATIONAL_MATRIX_AXIS
    value = Fraction(1, int("9" * MAX_INPUT_SCALAR_DIGITS))
    matrix = SparseRationalMatrix(
        row_count=order,
        column_count=order,
        entries=tuple(_entry(index, index, value) for index in range(order)),
    )

    result = rank_result(matrix)

    assert result.rank == order
    assert result.pivot_columns == tuple(range(order))
    assert result.matrix is matrix


def test_sparse_rank_accepts_a_max_axis_partial_permutation() -> None:
    axis = MAX_SPARSE_RATIONAL_MATRIX_AXIS
    entry_count = axis // 2
    matrix = SparseRationalMatrix(
        row_count=axis,
        column_count=axis,
        entries=tuple(_entry(row, axis - 1 - row) for row in range(entry_count)),
    )

    result = rank_result(matrix)

    assert result.rank == entry_count
    assert result.pivot_columns == tuple(range(entry_count, axis))
    assert result.matrix is matrix


def test_sparse_rank_rejects_a_connected_component_above_the_support_bound() -> None:
    order = isqrt(MAX_SPARSE_RANK_INTERMEDIATE_CELLS) + 1
    matrix = _connected_fill_matrix(order)

    with pytest.raises(OperationDomainValidationError, match="component-support bound"):
        rank_result(matrix)


def test_sparse_rank_charges_every_near_envelope_kernel_primitive() -> None:
    order = 464
    matrix = _connected_fill_matrix(order)
    executed = {"coordinate_load": 0, "gauss_jordan": 0}
    original_kernel = operations._sympy_sparse_rank_pivots

    def counted_kernel(plan: Any) -> tuple[int, ...]:
        for component in plan.components:
            executed["coordinate_load"] += (
                len(component.entries) * component.scalar_digits
            )
            executed["gauss_jordan"] += (
                len(component.rows)
                * len(component.columns)
                * min(len(component.rows), len(component.columns))
                * component.scalar_digits
            )
        return original_kernel(plan)

    with patch.object(
        operations, "_sympy_sparse_rank_pivots", side_effect=counted_kernel
    ):
        result = rank_result(matrix)

    charged = {
        "coordinate_load": len(matrix.entries),
        "gauss_jordan": order**3,
    }
    assert _connected_fill_work(order) == 99_899_662
    assert _connected_fill_work(order) <= MAX_EXACT_LINEAR_MATRIX_WORK
    assert result.rank == order
    assert result.pivot_columns == tuple(range(order))
    assert_charged_work_parity(charged=charged, executed=executed)


def test_sparse_rank_rejects_the_first_connected_order_above_the_flint_work_bound() -> (
    None
):
    order = 1
    while _connected_fill_work(order) <= MAX_FLINT_SPARSE_RANK_WORK:
        order += 1
    matrix = _connected_fill_matrix(order)

    with pytest.raises(OperationDomainValidationError, match="scalar-work budget"):
        rank_result(matrix)
    assert _connected_fill_work(order - 1) <= MAX_FLINT_SPARSE_RANK_WORK
    assert _connected_fill_work(order) > MAX_FLINT_SPARSE_RANK_WORK


def test_sparse_rank_uses_flint_for_a_dense_support_crossover() -> None:
    rows = 518
    columns = 298
    matrix = SparseRationalMatrix(
        row_count=rows,
        column_count=columns,
        entries=(
            *(_entry(0, column, Fraction(1000)) for column in range(columns)),
            *(_entry(row, 0, Fraction(1000)) for row in range(1, rows)),
        ),
    )

    result = rank_result(matrix)

    assert result.rank == 2
    assert result.pivot_columns == (0, 1)


def test_sparse_rank_rejects_above_the_intermediate_height_bound() -> None:
    scalar_limit = 10**MAX_INPUT_SCALAR_DIGITS - 1
    denominators: list[int] = []
    for prime in islice(primerange(2, 10_000), 130):
        denominator = 1
        while denominator * prime <= scalar_limit:
            denominator *= prime
        denominators.append(denominator)
    matrix = SparseRationalMatrix(
        row_count=1,
        column_count=len(denominators),
        entries=tuple(
            _entry(0, column, Fraction(1, denominator))
            for column, denominator in enumerate(denominators)
        ),
    )

    with pytest.raises(OperationDomainValidationError, match="intermediate-height"):
        rank_result(matrix)


@pytest.mark.scale
def test_sparse_rank_accepts_the_structurally_bounded_source() -> None:
    value = CanonicalRational(
        num="1" + "0" * (MAX_INPUT_SCALAR_DIGITS - 1),
        den="9" * MAX_INPUT_SCALAR_DIGITS,
    )
    matrix = SparseRationalMatrix(
        row_count=4,
        column_count=MAX_SPARSE_RATIONAL_MATRIX_AXIS,
        entries=tuple(
            SparseRationalMatrixEntry(row=row, column=column, value=value)
            for row in range(4)
            for column in range(MAX_SPARSE_RATIONAL_MATRIX_AXIS)
        ),
    )

    result = rank_result(matrix)

    assert result.rank == 1


def test_sparse_rank_result_rejects_rank_above_a_source_axis() -> None:
    with pytest.raises(
        ValidationError, match="cannot exceed either source matrix axis"
    ):
        MatrixRankResult(
            matrix=SparseRationalMatrix(
                row_count=1,
                column_count=2,
                entries=(_entry(0, 0), _entry(0, 1)),
            ),
            rank=2,
            pivot_columns=(0, 1),
        )
