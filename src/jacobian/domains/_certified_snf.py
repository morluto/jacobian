"""Bounded elementary-operation Smith reduction used by trusted producers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

from jacobian.contracts.certified_snf import (
    CertifiedIntegerMatrix,
    SmithNormalFormCertificate,
)

Matrix = list[list[int]]


@dataclass(frozen=True, slots=True)
class SmithReduction:
    source: Matrix
    diagonal: Matrix
    left: Matrix
    right: Matrix
    rank: int
    invariant_factors: tuple[int, ...]
    left_determinant: int
    right_determinant: int


def zero_matrix(rows: int, columns: int) -> Matrix:
    return [[0 for _ in range(columns)] for _ in range(rows)]


def identity_matrix(size: int) -> Matrix:
    result = zero_matrix(size, size)
    for index in range(size):
        result[index][index] = 1
    return result


def matrix_shape(matrix: Matrix, *, columns_if_empty: int = 0) -> tuple[int, int]:
    return len(matrix), len(matrix[0]) if matrix else columns_if_empty


def matrix_multiply(
    left: Matrix,
    right: Matrix,
    *,
    right_columns_if_empty: int = 0,
) -> Matrix:
    left_rows = len(left)
    middle = len(left[0]) if left else len(right)
    right_rows = len(right)
    right_columns = len(right[0]) if right else right_columns_if_empty
    if middle != right_rows:
        raise ValueError("integer matrices are not composable")
    return [
        [
            sum(left[row][index] * right[index][column] for index in range(middle))
            for column in range(right_columns)
        ]
        for row in range(left_rows)
    ]


def matrix_vector_multiply(matrix: Matrix, vector: list[int]) -> list[int]:
    if matrix and len(matrix[0]) != len(vector):
        raise ValueError("integer matrix and vector are not composable")
    return [
        sum(value * vector[index] for index, value in enumerate(row)) for row in matrix
    ]


def matrix_columns(matrix: Matrix, start: int = 0) -> Matrix:
    if not matrix:
        return []
    return [
        [matrix[row][column] for column in range(start, len(matrix[0]))]
        for row in range(len(matrix))
    ]


def determinant(matrix: Matrix) -> int:
    """Exact fraction-free Bareiss determinant."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 0:
        return 1
    work = [row[:] for row in matrix]
    sign = 1
    previous = 1
    for pivot_index in range(size - 1):
        selected = next(
            (row for row in range(pivot_index, size) if work[row][pivot_index] != 0),
            None,
        )
        if selected is None:
            return 0
        if selected != pivot_index:
            work[pivot_index], work[selected] = work[selected], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, size):
            for column in range(pivot_index + 1, size):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                if numerator % previous:
                    raise ArithmeticError("Bareiss division was not exact")
                work[row][column] = numerator // previous
        previous = pivot
    return sign * work[-1][-1]


def inverse_unimodular(matrix: Matrix) -> Matrix:
    """Invert a unimodular integer matrix by exact Gauss-Jordan elimination."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("unimodular inverse requires a square matrix")
    if size == 0:
        return []
    augmented = [
        [*row, *(1 if row_index == column else 0 for column in range(size))]
        for row_index, row in enumerate(matrix)
    ]
    for column in range(size):
        selected = next(
            (row for row in range(column, size) if augmented[row][column] != 0),
            None,
        )
        if selected is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[selected] = (
            augmented[selected],
            augmented[column],
        )
        while abs(augmented[column][column]) != 1:
            selected = next(
                (row for row in range(column + 1, size) if augmented[row][column] != 0),
                None,
            )
            if selected is None:
                raise ValueError("matrix is not unimodular")
            quotient = augmented[column][column] // augmented[selected][column]
            augmented[column] = [
                value - quotient * other
                for value, other in zip(
                    augmented[column], augmented[selected], strict=True
                )
            ]
            augmented[column], augmented[selected] = (
                augmented[selected],
                augmented[column],
            )
        pivot = augmented[column][column]
        if pivot == -1:
            augmented[column] = [-value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]
    return [row[size:] for row in augmented]


def _swap_rows(matrix: Matrix, left: int, right: int) -> None:
    matrix[left], matrix[right] = matrix[right], matrix[left]


def _swap_columns(matrix: Matrix, left: int, right: int) -> None:
    for row in matrix:
        row[left], row[right] = row[right], row[left]


def _add_row_multiple(
    matrix: Matrix, target: int, source: int, multiplier: int
) -> None:
    matrix[target] = [
        value + multiplier * other
        for value, other in zip(matrix[target], matrix[source], strict=True)
    ]


def _add_column_multiple(
    matrix: Matrix, target: int, source: int, multiplier: int
) -> None:
    for row in matrix:
        row[target] += multiplier * row[source]


def _negate_row(matrix: Matrix, row: int) -> None:
    matrix[row] = [-value for value in matrix[row]]


def smith_reduce(
    source: Matrix,
    *,
    row_count: int | None = None,
    column_count: int | None = None,
) -> SmithReduction:
    """Return a canonical Smith diagonal and explicit elementary transformations."""

    rows = len(source) if row_count is None else row_count
    columns = (
        (len(source[0]) if source else 0) if column_count is None else column_count
    )
    if len(source) != rows or any(len(row) != columns for row in source):
        raise ValueError("source entries do not match the declared matrix shape")
    original = [row[:] for row in source]
    matrix = [row[:] for row in source]
    left = identity_matrix(rows)
    right = identity_matrix(columns)
    diagonal_count = min(rows, columns)

    for pivot in range(diagonal_count):
        selected = min(
            (
                (abs(matrix[row][column]), row, column)
                for row in range(pivot, rows)
                for column in range(pivot, columns)
                if matrix[row][column] != 0
            ),
            default=None,
        )
        if selected is None:
            break
        _, selected_row, selected_column = selected
        if selected_row != pivot:
            _swap_rows(matrix, pivot, selected_row)
            _swap_rows(left, pivot, selected_row)
        if selected_column != pivot:
            _swap_columns(matrix, pivot, selected_column)
            _swap_columns(right, pivot, selected_column)

        while True:
            changed = False
            for row in range(pivot + 1, rows):
                while matrix[row][pivot] != 0:
                    quotient = matrix[row][pivot] // matrix[pivot][pivot]
                    _add_row_multiple(matrix, row, pivot, -quotient)
                    _add_row_multiple(left, row, pivot, -quotient)
                    if matrix[row][pivot] != 0 and abs(matrix[row][pivot]) < abs(
                        matrix[pivot][pivot]
                    ):
                        _swap_rows(matrix, row, pivot)
                        _swap_rows(left, row, pivot)
                    changed = True
            for column in range(pivot + 1, columns):
                while matrix[pivot][column] != 0:
                    quotient = matrix[pivot][column] // matrix[pivot][pivot]
                    _add_column_multiple(matrix, column, pivot, -quotient)
                    _add_column_multiple(right, column, pivot, -quotient)
                    if matrix[pivot][column] != 0 and abs(matrix[pivot][column]) < abs(
                        matrix[pivot][pivot]
                    ):
                        _swap_columns(matrix, column, pivot)
                        _swap_columns(right, column, pivot)
                    changed = True
            offender = next(
                (
                    (row, column)
                    for row in range(pivot + 1, rows)
                    for column in range(pivot + 1, columns)
                    if matrix[row][column] % matrix[pivot][pivot]
                ),
                None,
            )
            if offender is not None:
                _add_row_multiple(matrix, pivot, offender[0], 1)
                _add_row_multiple(left, pivot, offender[0], 1)
                changed = True
                continue
            if not changed:
                break
            if all(matrix[row][pivot] == 0 for row in range(pivot + 1, rows)) and all(
                matrix[pivot][column] == 0 for column in range(pivot + 1, columns)
            ):
                break
        if matrix[pivot][pivot] < 0:
            _negate_row(matrix, pivot)
            _negate_row(left, pivot)

    factors = tuple(
        matrix[index][index]
        for index in range(diagonal_count)
        if matrix[index][index] != 0
    )
    if any(value <= 0 for value in factors) or any(
        right_factor % left_factor for left_factor, right_factor in pairwise(factors)
    ):
        raise ArithmeticError("Smith reduction did not produce a canonical diagonal")
    if matrix_multiply(matrix_multiply(left, original), right) != matrix:
        raise ArithmeticError("Smith transformations do not bind the source")
    left_determinant = determinant(left)
    right_determinant = determinant(right)
    if abs(left_determinant) != 1 or abs(right_determinant) != 1:
        raise ArithmeticError("Smith transformations are not unimodular")
    return SmithReduction(
        source=original,
        diagonal=matrix,
        left=left,
        right=right,
        rank=len(factors),
        invariant_factors=factors,
        left_determinant=left_determinant,
        right_determinant=right_determinant,
    )


def _contract_matrix(
    entries: Matrix,
    *,
    rows: int,
    columns: int,
) -> CertifiedIntegerMatrix:
    return CertifiedIntegerMatrix(
        row_count=rows,
        column_count=columns,
        entries=tuple(tuple(str(value) for value in row) for row in entries),
    )


def certificate_from_reduction(
    reduction: SmithReduction,
) -> SmithNormalFormCertificate:
    rows = len(reduction.source)
    columns = len(reduction.source[0]) if reduction.source else len(reduction.right)
    left_determinant: Literal["-1", "1"] = (
        "1" if reduction.left_determinant == 1 else "-1"
    )
    right_determinant: Literal["-1", "1"] = (
        "1" if reduction.right_determinant == 1 else "-1"
    )
    return SmithNormalFormCertificate(
        source=_contract_matrix(reduction.source, rows=rows, columns=columns),
        diagonal=_contract_matrix(reduction.diagonal, rows=rows, columns=columns),
        left_transformation=_contract_matrix(
            reduction.left,
            rows=rows,
            columns=rows,
        ),
        right_transformation=_contract_matrix(
            reduction.right,
            rows=columns,
            columns=columns,
        ),
        rank=reduction.rank,
        invariant_factors=tuple(str(value) for value in reduction.invariant_factors),
        left_determinant=left_determinant,
        right_determinant=right_determinant,
    )


__all__ = [
    "Matrix",
    "SmithReduction",
    "certificate_from_reduction",
    "determinant",
    "identity_matrix",
    "inverse_unimodular",
    "matrix_columns",
    "matrix_multiply",
    "matrix_shape",
    "matrix_vector_multiply",
    "smith_reduce",
    "zero_matrix",
]
