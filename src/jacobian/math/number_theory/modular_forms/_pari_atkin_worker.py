"""One-shot exact PARI worker for bounded Atkin-Lehner action matrices."""

from __future__ import annotations

import hashlib
import sys
from fractions import Fraction
from math import gcd
from typing import Any

from jacobian._worker_protocol import encode_worker_result_frame
from jacobian.canonical import format_canonical_integer, loads_strict_json

_MAX_LEVEL = 10_000
_MAX_WEIGHT = 120
_MAX_PRECISION = 128
_MAX_DIMENSION = 32
_MATRIX_DIGITS = 512
_INPUT_DIGITS = 4096
_INTERNAL_DIGITS = 10_000_000


def _int(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("Atkin-Lehner worker input is outside its admitted range")
    return value


def _digits(value: Fraction) -> int:
    return max(len(str(abs(value.numerator))), len(str(value.denominator)))


def _fraction(value: Any) -> Fraction:
    numerator, denominator = int(value.numerator()), int(value.denominator())
    if denominator <= 0:
        raise RuntimeError("PARI returned a rational with nonpositive denominator")
    return Fraction(numerator, denominator)


def _parse_pair(value: object, max_digits: int) -> Fraction:
    if (
        type(value) is not list
        or len(value) != 2
        or any(type(x) is not str for x in value)
    ):
        raise ValueError("Atkin-Lehner worker rational pair is malformed")
    rational = Fraction(int(value[0]), int(value[1]))
    if (str(rational.numerator), str(rational.denominator)) != tuple(value):
        raise ValueError("Atkin-Lehner worker rational is not canonical")
    if _digits(rational) > max_digits:
        raise ValueError("Atkin-Lehner worker rational exceeds its admitted height")
    return rational


def _pair(value: Fraction) -> list[str]:
    return [
        format_canonical_integer(value.numerator),
        format_canonical_integer(value.denominator),
    ]


def _check_matrix(matrix: list[list[Fraction]], max_digits: int) -> None:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise RuntimeError("Atkin-Lehner matrix is not square")
    if any(_digits(value) > max_digits for row in matrix for value in row):
        raise RuntimeError("Atkin-Lehner matrix entry exceeds its height envelope")


def _multiply(
    left: list[list[Fraction]], right: list[list[Fraction]], max_digits: int
) -> list[list[Fraction]]:
    size = len(left)
    output = []
    for row in range(size):
        output_row = []
        for column in range(size):
            value = sum(
                (left[row][index] * right[index][column] for index in range(size)),
                Fraction(0),
            )
            if _digits(value) > max_digits:
                raise RuntimeError(
                    "Atkin-Lehner matrix intermediate exceeds its height cap"
                )
            output_row.append(value)
        output.append(output_row)
    return output


def _inverse(matrix: list[list[Fraction]], max_digits: int) -> list[list[Fraction]]:
    size = len(matrix)
    rows = [
        [*matrix[row], *(Fraction(int(row == column)) for column in range(size))]
        for row in range(size)
    ]
    for column in range(size):
        pivot = next((row for row in range(column, size) if rows[row][column]), None)
        if pivot is None:
            raise RuntimeError("Atkin-Lehner basis change is singular")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for row in range(size):
            if row == column or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(rows[row], rows[column], strict=True)
            ]
        if any(_digits(value) > max_digits for row in rows for value in row):
            raise RuntimeError("Atkin-Lehner basis inverse exceeds its height cap")
    return [row[size:] for row in rows]


def _action_matrix(request: dict[str, object]) -> list[list[Fraction]]:
    import cypari

    pari = cypari.pari
    level = int(request["level"])
    weight = int(request["weight"])
    kind = request["kind"]
    divisor = int(request["divisor"])
    basis_vectors = request["basis_vectors"]
    dimension = len(basis_vectors)
    if dimension == 0:
        return []
    mf = pari.mfinit([level, weight], 4 if kind == "M" else 1)
    if int(pari.mfdim(mf)) != dimension:
        raise RuntimeError("PARI dimension disagrees with the admitted space")

    # T columns are PARI's coordinates of Jacobian's canonical basis vectors.
    change = [[Fraction(0) for _ in range(dimension)] for _ in range(dimension)]
    for column, raw_vector in enumerate(basis_vectors):
        prefix = [pari(value.numerator) / value.denominator for value in raw_vector]
        backend_coordinates = pari.mftobasis(mf, prefix)
        if len(backend_coordinates) != dimension:
            raise RuntimeError("PARI did not recover unique basis coordinates")
        for row, value in enumerate(backend_coordinates):
            change[row][column] = _fraction(value)
    _check_matrix(change, _MATRIX_DIGITS)

    atkin = pari.mfatkininit(mf, divisor)
    if int(atkin[0]) != 0:
        raise RuntimeError("Atkin-Lehner backend returned a different target space")
    raw_operator = atkin[1]
    scale = _fraction(atkin[2])
    size_rows, size_columns = (int(x) for x in raw_operator.matsize())
    if size_rows != dimension or size_columns != dimension:
        raise RuntimeError("PARI Atkin-Lehner matrix has an unexpected shape")
    operator = [
        [_fraction(raw_operator[row, column]) / scale for column in range(dimension)]
        for row in range(dimension)
    ]
    _check_matrix(operator, _MATRIX_DIGITS)

    inverse = _inverse(change, _INTERNAL_DIGITS)
    left_product = _multiply(inverse, operator, _INTERNAL_DIGITS)
    result = _multiply(left_product, change, _MATRIX_DIGITS)
    _check_matrix(result, _MATRIX_DIGITS)
    return result


def main() -> int:
    input_bytes = sys.stdin.buffer.read()
    request = loads_strict_json(input_bytes)
    expected_keys = {"level", "weight", "kind", "divisor", "precision", "basis_vectors"}
    if not isinstance(request, dict) or set(request) != expected_keys:
        raise ValueError("Atkin-Lehner worker request has an invalid shape")
    level = _int(request["level"], 1, _MAX_LEVEL)
    _int(request["weight"], 0, _MAX_WEIGHT)
    divisor = _int(request["divisor"], 1, level)
    precision = _int(request["precision"], 1, _MAX_PRECISION)
    if type(request["kind"]) is not str or request["kind"] not in ("M", "S"):
        raise ValueError("Atkin-Lehner worker kind must be M or S")
    if level % divisor or gcd(divisor, level // divisor) != 1:
        raise ValueError("Atkin-Lehner divisor must be exact")
    raw_basis = request["basis_vectors"]
    if type(raw_basis) is not list or len(raw_basis) > _MAX_DIMENSION:
        raise ValueError("Atkin-Lehner basis has an invalid dimension")
    basis_vectors = []
    for raw_vector in raw_basis:
        if type(raw_vector) is not list or len(raw_vector) != precision:
            raise ValueError("Atkin-Lehner basis vector has an invalid precision")
        basis_vectors.append(
            [_parse_pair(value, _INPUT_DIGITS) for value in raw_vector]
        )
    request["basis_vectors"] = basis_vectors
    matrix = _action_matrix(request)
    output = [[_pair(value) for value in row] for row in matrix]
    digest = hashlib.sha256(input_bytes).hexdigest()
    sys.stdout.buffer.write(
        encode_worker_result_frame(
            {"kind": "complete", "matrix": output, "request_digest": digest}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
