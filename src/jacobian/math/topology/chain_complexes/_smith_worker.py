"""One-shot SymPy Smith-decomposition worker for integral homology."""

from __future__ import annotations

import hashlib
import sys
from itertools import pairwise
from typing import Any

from jacobian.canonical import (
    CanonicalLimits,
    encode_strict_json,
    format_canonical_integer,
    loads_strict_json,
    parse_canonical_integer,
)


def _matrix_entries(matrix: Any, *, rows: int, columns: int) -> list[list[int]]:
    return [
        [_decode_integer(matrix[row, column]) for column in range(columns)]
        for row in range(rows)
    ]


def _decode_integer(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_canonical_integer(value)
    try:
        # SymPy Integer exposes an exact Python-int conversion without first
        # formatting through Python's bounded decimal codec.
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("worker matrix entry is not an integer") from exc


def _encode_matrix(matrix: list[list[int]]) -> list[list[str]]:
    return [[format_canonical_integer(value) for value in row] for row in matrix]


def _inverse_unimodular(matrix: list[list[int]]) -> list[list[int]]:
    """Compute the inverse needed by homology coordinates without rechecking it."""

    size = len(matrix)
    if size == 0:
        return []

    from sympy import ZZ
    from sympy.polys.matrices import DomainMatrix

    domain = DomainMatrix(
        [[ZZ(value) for value in row] for row in matrix],
        (size, size),
        ZZ,
    )
    numerator, denominator = domain.inv_den(method="rref")
    if denominator == -ZZ.one:
        numerator = -numerator
    elif denominator != ZZ.one:
        raise ArithmeticError("Smith transformation inverse is not integral")
    return [[int(value) for value in row] for row in numerator.to_list()]


def _smith_projection(
    source: list[list[int]], *, rows: int, columns: int
) -> dict[str, Any]:
    """Compute one owner-private projection with maintained SymPy kernels."""

    import sympy
    from sympy.matrices.normalforms import smith_normal_decomp

    source_matrix = (
        sympy.Matrix([[_decode_integer(value) for value in row] for row in source])
        if rows and columns
        else sympy.Matrix(rows, columns, [])
    )
    diagonal, left, right = smith_normal_decomp(source_matrix, domain=sympy.ZZ)
    diagonal_entries = _matrix_entries(diagonal, rows=rows, columns=columns)
    left_entries = _matrix_entries(left, rows=rows, columns=rows)
    right_entries = _matrix_entries(right, rows=columns, columns=columns)
    diagonal_values = tuple(
        diagonal_entries[index][index] for index in range(min(rows, columns))
    )
    factors = tuple(value for value in diagonal_values if value != 0)
    if (
        diagonal_values[: len(factors)] != factors
        or any(value != 0 for value in diagonal_values[len(factors) :])
        or any(value <= 0 for value in factors)
        or any(
            right_factor % left_factor
            for left_factor, right_factor in pairwise(factors)
        )
        or any(
            diagonal_entries[row][column] != 0
            for row in range(rows)
            for column in range(columns)
            if row != column
        )
    ):
        raise ArithmeticError("SymPy returned a noncanonical Smith diagonal")
    return {
        "diagonal": diagonal_entries,
        "left": left_entries,
        "right": right_entries,
        "rank": len(factors),
        "invariant_factors": factors,
        "left_determinant": int(left.det()),
        "right_determinant": int(right.det()),
        "left_inverse": _inverse_unimodular(left_entries),
        "right_inverse": _inverse_unimodular(right_entries),
    }


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    payload: dict[str, Any] = loads_strict_json(input_bytes)
    source = payload["matrix"]
    rows = payload["row_count"]
    columns = payload["column_count"]
    output_limit = payload["output_limit"]
    if (
        not isinstance(output_limit, int)
        or isinstance(output_limit, bool)
        or output_limit < 1
    ):
        raise ValueError("worker output limit is invalid")

    projection = _smith_projection(source, rows=rows, columns=columns)
    response = {
        "request_digest": hashlib.sha256(input_bytes).hexdigest(),
        "diagonal": _encode_matrix(projection["diagonal"]),
        "left": _encode_matrix(projection["left"]),
        "right": _encode_matrix(projection["right"]),
        "rank": projection["rank"],
        "invariant_factors": [
            format_canonical_integer(value) for value in projection["invariant_factors"]
        ],
        "left_determinant": projection["left_determinant"],
        "right_determinant": projection["right_determinant"],
        "left_inverse": _encode_matrix(projection["left_inverse"]),
        "right_inverse": _encode_matrix(projection["right_inverse"]),
    }
    sys.stdout.buffer.write(
        encode_strict_json(
            response,
            limits=CanonicalLimits(max_output_bytes=output_limit),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
