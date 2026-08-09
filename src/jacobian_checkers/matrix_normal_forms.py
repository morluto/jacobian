"""Independent exact replay for retained integer row-HNF certificates."""

from __future__ import annotations

import re
from typing import Any

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.matrices import MAX_MATRIX_SCALAR_DIGITS
from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
MAX_MATRIX_DIMENSION = 32
MAX_INTEGER_DIGITS = MAX_MATRIX_SCALAR_DIGITS


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _integer(value: object) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > MAX_INTEGER_DIGITS
    ):
        raise ValueError("matrix entry is not a bounded canonical integer")
    result = parse_canonical_integer(value)
    if format_canonical_integer(result) != value:
        raise ValueError("matrix entry is not canonical")
    return result


def _matrix(payload: object) -> list[list[int]]:
    if not isinstance(payload, dict) or set(payload) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise ValueError("integer matrix has an invalid shape")
    if payload["matrix_schema_version"] != "1" or payload["domain"] != "ZZ":
        raise ValueError("integer matrix uses unsupported semantics")
    entries = payload["entries"]
    if (
        not isinstance(entries, list)
        or not 1 <= len(entries) <= MAX_MATRIX_DIMENSION
        or not isinstance(entries[0], list)
        or not 1 <= len(entries[0]) <= MAX_MATRIX_DIMENSION
        or any(
            not isinstance(row, list) or len(row) != len(entries[0]) for row in entries
        )
    ):
        raise ValueError("integer matrix dimensions are malformed")
    return [[_integer(value) for value in row] for row in entries]


def _multiply(
    left: list[list[int]],
    right: list[list[int]],
) -> list[list[int]]:
    return [
        [
            sum(
                left_value * right[row][column]
                for row, left_value in enumerate(left_row)
            )
            for column in range(len(right[0]))
        ]
        for left_row in left
    ]


def _determinant_bareiss(matrix: list[list[int]]) -> int:
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("determinant requires a square matrix")
    if size == 1:
        return matrix[0][0]
    work = [row[:] for row in matrix]
    sign = 1
    previous_pivot = 1
    for column in range(size - 1):
        pivot_row = next(
            (row for row in range(column, size) if work[row][column] != 0),
            None,
        )
        if pivot_row is None:
            return 0
        if pivot_row != column:
            work[column], work[pivot_row] = work[pivot_row], work[column]
            sign = -sign
        pivot = work[column][column]
        for row in range(column + 1, size):
            for target_column in range(column + 1, size):
                numerator = (
                    work[row][target_column] * pivot
                    - work[row][column] * work[column][target_column]
                )
                quotient, remainder = divmod(numerator, previous_pivot)
                if remainder != 0:
                    raise ValueError("fraction-free determinant division failed")
                work[row][target_column] = quotient
            work[row][column] = 0
        previous_pivot = pivot
    return sign * work[-1][-1]


def _is_row_hnf(matrix: list[list[int]]) -> bool:
    last_nonzero_row = -1
    for row, values in enumerate(matrix):
        if any(value != 0 for value in values):
            last_nonzero_row = row
    previous_pivot_column = -1
    for row in range(last_nonzero_row + 1):
        pivot_column = next(
            (column for column, value in enumerate(matrix[row]) if value != 0),
            None,
        )
        if pivot_column is None or pivot_column <= previous_pivot_column:
            return False
        pivot = matrix[row][pivot_column]
        if pivot < 0:
            return False
        if any(
            matrix[above][pivot_column] < 0 or matrix[above][pivot_column] >= pivot
            for above in range(row)
        ):
            return False
        previous_pivot_column = pivot_column
    return all(
        all(value == 0 for value in matrix[row])
        for row in range(last_nonzero_row + 1, len(matrix))
    )


def _check_hnf_math(
    transformation: list[list[int]],
    source: list[list[int]],
    normal_form: list[list[int]],
) -> str | None:
    if _multiply(transformation, source) != normal_form:
        return "the proposed exact relation H = U A does not hold"
    if abs(_determinant_bareiss(transformation)) != 1:
        return "the proposed left transformation is not unimodular"
    if not _is_row_hnf(normal_form):
        return "the candidate does not satisfy FLINT row-HNF conditions"
    return None


def _check_materialized_operation_request(request: dict[str, Any]) -> dict[str, Any]:
    claim, candidate = bound_request(
        request,
        operation_id="matrix.normal_form.hermite.materialize",
        witness_format="matrix.normal_form.hermite",
    )
    source = _matrix(claim["matrix"])
    if set(candidate) != {
        "result_schema_version",
        "normal_form",
        "transformation",
        "method",
        "backend",
        "backend_version",
        "flint_library_version",
    }:
        return _reject("HNF result is malformed")
    if candidate["method"] != "ROW_HNF_LEFT_UNIMODULAR_TRANSFORM":
        return _reject("HNF result uses unsupported semantics")
    normal_form = _matrix(candidate["normal_form"])
    transformation = _matrix(candidate["transformation"])
    if (
        len(normal_form) != len(source)
        or len(normal_form[0]) != len(source[0])
        or len(transformation) != len(source)
        or len(transformation[0]) != len(source)
    ):
        return _reject("HNF result dimensions are malformed")
    error = _check_hnf_math(transformation, source, normal_form)
    if error is not None:
        return _reject(error)
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": "checked H = U A and row-HNF conditions independently",
    }


def check_hermite_normal_form(request: dict[str, Any]) -> dict[str, Any]:
    """Accept only the v1 retained HNF result certificate."""

    try:
        return _check_materialized_operation_request(request)
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed HNF request")
