"""Independent Smith-certificate replay with no producer or contract imports."""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from jacobian_checkers.bound_artifacts import bound_request

_INTEGER = re.compile(r"^(?:0|-?[1-9][0-9]*)$")
_MAX_DIMENSION = 32
_MAX_INPUT_DIMENSION = 16
_MAX_INPUT_DIGITS = 32
_MAX_OUTPUT_DIGITS = 32_768
_DECIMAL_CHUNK_DIGITS = 9
_DECIMAL_CHUNK_BASE = 1_000_000_000


@dataclass(frozen=True, slots=True)
class ParsedMatrix:
    rows: int
    columns: int
    entries: list[list[int]]


@dataclass(frozen=True, slots=True)
class ParsedSmithCertificate:
    source: ParsedMatrix
    diagonal: ParsedMatrix
    left: ParsedMatrix
    right: ParsedMatrix
    rank: int
    factors: tuple[int, ...]


def _reject(detail: str) -> dict[str, Any]:
    return {
        "accepted": False,
        "conclusion": "UNKNOWN",
        "arithmetic": "EXACT_INTEGER",
        "method": "DIRECT_WITNESS",
        "coverage": "NOT_APPLICABLE",
        "detail": detail,
    }


def _accept(detail: str) -> dict[str, Any]:
    return {
        "accepted": True,
        "conclusion": "TRUE",
        "arithmetic": "EXACT_INTEGER",
        "method": "EXHAUSTIVE_FINITE",
        "coverage": "EXHAUSTIVE",
        "detail": detail,
    }


def _strict_int(value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("integer lies outside the checker scope")
    return value


def _canonical_integer(value: object, *, maximum_digits: int) -> int:
    if (
        not isinstance(value, str)
        or _INTEGER.fullmatch(value) is None
        or len(value.lstrip("-")) > maximum_digits
    ):
        raise ValueError("integer encoding lies outside the checker scope")
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    first = len(digits) % _DECIMAL_CHUNK_DIGITS or _DECIMAL_CHUNK_DIGITS
    parsed = int(digits[:first])
    for offset in range(first, len(digits), _DECIMAL_CHUNK_DIGITS):
        parsed = parsed * _DECIMAL_CHUNK_BASE + int(
            digits[offset : offset + _DECIMAL_CHUNK_DIGITS]
        )
    return -parsed if negative else parsed


def parse_matrix(value: object, *, maximum_digits: int) -> ParsedMatrix:
    if not isinstance(value, dict) or set(value) != {
        "matrix_schema_version",
        "domain",
        "row_count",
        "column_count",
        "entries",
    }:
        raise ValueError("integer matrix payload is malformed")
    if value["matrix_schema_version"] != "1" or value["domain"] != "ZZ":
        raise ValueError("integer matrix semantics are unsupported")
    rows = _strict_int(value["row_count"], minimum=0, maximum=_MAX_DIMENSION)
    columns = _strict_int(value["column_count"], minimum=0, maximum=_MAX_DIMENSION)
    entries = value["entries"]
    if (
        not isinstance(entries, list)
        or len(entries) != rows
        or any(not isinstance(row, list) or len(row) != columns for row in entries)
    ):
        raise ValueError("integer matrix shape is malformed")
    return ParsedMatrix(
        rows=rows,
        columns=columns,
        entries=[
            [_canonical_integer(item, maximum_digits=maximum_digits) for item in row]
            for row in entries
        ],
    )


def _multiply(left: ParsedMatrix, right: ParsedMatrix) -> ParsedMatrix:
    if left.columns != right.rows:
        raise ValueError("integer matrices are not composable")
    return ParsedMatrix(
        rows=left.rows,
        columns=right.columns,
        entries=[
            [
                sum(
                    left.entries[row][middle] * right.entries[middle][column]
                    for middle in range(left.columns)
                )
                for column in range(right.columns)
            ]
            for row in range(left.rows)
        ],
    )


def _determinant(matrix: ParsedMatrix) -> int:
    if matrix.rows != matrix.columns:
        raise ValueError("determinant requires a square matrix")
    if matrix.rows == 0:
        return 1
    work = [row[:] for row in matrix.entries]
    sign = 1
    previous = 1
    for pivot_index in range(matrix.rows - 1):
        selected = next(
            (
                row
                for row in range(pivot_index, matrix.rows)
                if work[row][pivot_index] != 0
            ),
            None,
        )
        if selected is None:
            return 0
        if selected != pivot_index:
            work[pivot_index], work[selected] = work[selected], work[pivot_index]
            sign = -sign
        pivot = work[pivot_index][pivot_index]
        for row in range(pivot_index + 1, matrix.rows):
            for column in range(pivot_index + 1, matrix.rows):
                numerator = (
                    work[row][column] * pivot
                    - work[row][pivot_index] * work[pivot_index][column]
                )
                if numerator % previous:
                    raise ValueError("fraction-free determinant replay failed")
                work[row][column] = numerator // previous
        previous = pivot
    return sign * work[-1][-1]


def validate_certificate(value: object) -> ParsedSmithCertificate:
    if not isinstance(value, dict) or set(value) != {
        "certificate_schema_version",
        "source",
        "diagonal",
        "left_transformation",
        "right_transformation",
        "rank",
        "invariant_factors",
        "left_determinant",
        "right_determinant",
        "relation",
        "transformation_scope",
        "convention",
    }:
        raise ValueError("Smith certificate payload is malformed")
    if (
        value["certificate_schema_version"] != "1"
        or value["relation"] != "DIAGONAL_EQUALS_LEFT_TIMES_SOURCE_TIMES_RIGHT"
        or value["transformation_scope"] != "FULL_BASIS_BOTH_SIDES"
        or value["convention"] != "POSITIVE_DIVISIBILITY_DIAGONAL"
    ):
        raise ValueError("Smith certificate semantics are unsupported")
    source = parse_matrix(value["source"], maximum_digits=_MAX_OUTPUT_DIGITS)
    diagonal = parse_matrix(value["diagonal"], maximum_digits=_MAX_OUTPUT_DIGITS)
    left = parse_matrix(value["left_transformation"], maximum_digits=_MAX_OUTPUT_DIGITS)
    right = parse_matrix(
        value["right_transformation"], maximum_digits=_MAX_OUTPUT_DIGITS
    )
    if (
        (diagonal.rows, diagonal.columns) != (source.rows, source.columns)
        or (left.rows, left.columns) != (source.rows, source.rows)
        or (right.rows, right.columns) != (source.columns, source.columns)
    ):
        raise ValueError("Smith certificate matrix shapes are incompatible")
    rank = _strict_int(
        value["rank"],
        minimum=0,
        maximum=min(source.rows, source.columns),
    )
    raw_factors = value["invariant_factors"]
    if not isinstance(raw_factors, list):
        raise ValueError("Smith invariant factors are malformed")
    factors = tuple(
        _canonical_integer(item, maximum_digits=_MAX_OUTPUT_DIGITS)
        for item in raw_factors
    )
    diagonal_values = tuple(
        diagonal.entries[index][index]
        for index in range(min(source.rows, source.columns))
    )
    if (
        rank != len(factors)
        or any(value <= 0 for value in factors)
        or diagonal_values[:rank] != factors
        or any(value != 0 for value in diagonal_values[rank:])
        or any(
            diagonal.entries[row][column] != 0
            for row in range(diagonal.rows)
            for column in range(diagonal.columns)
            if row != column
        )
        or any(
            right_factor % left_factor
            for left_factor, right_factor in pairwise(factors)
        )
    ):
        raise ValueError("Smith diagonal is not canonical")
    left_determinant = _determinant(left)
    right_determinant = _determinant(right)
    if (
        abs(left_determinant) != 1
        or abs(right_determinant) != 1
        or value["left_determinant"] != str(left_determinant)
        or value["right_determinant"] != str(right_determinant)
    ):
        raise ValueError("Smith transformations are not unimodular")
    replayed = _multiply(_multiply(left, source), right)
    if replayed.entries != diagonal.entries:
        raise ValueError("Smith transformation relation does not hold")
    return ParsedSmithCertificate(
        source=source,
        diagonal=diagonal,
        left=left,
        right=right,
        rank=rank,
        factors=factors,
    )


def check_certified_smith_normal_form(request: object) -> dict[str, Any]:
    try:
        source, result = bound_request(
            request,
            operation_id="matrix.normal_form.smith.certified.compute",
            witness_format="matrix.smith-normal-form.transformation-certificate-v1",
        )
        if not isinstance(source, dict) or set(source) != {"matrix"}:
            raise ValueError("certified Smith request is malformed")
        source_matrix = parse_matrix(source["matrix"], maximum_digits=_MAX_INPUT_DIGITS)
        if (
            not 1 <= source_matrix.rows <= _MAX_INPUT_DIMENSION
            or not 1 <= source_matrix.columns <= _MAX_INPUT_DIMENSION
        ):
            raise ValueError("certified Smith request lies outside checker scope")
        if not isinstance(result, dict) or set(result) != {
            "certificate",
            "exactness",
            "determinism",
            "backend",
            "backend_version",
            "completeness",
            "verification",
        }:
            raise ValueError("certified Smith result is malformed")
        if (
            result["exactness"] != "EXACT_INTEGER"
            or result["determinism"] != "DETERMINISTIC"
            or result["backend"] != "jacobian-sympy-smith-normal-decomposition"
            or result["backend_version"] != "1"
            or result["completeness"] != "FULL_MATRIX_TRANSFORMATIONS"
            or result["verification"] != "UNVERIFIED"
        ):
            raise ValueError("certified Smith result metadata is unsupported")
        certificate = validate_certificate(result["certificate"])
        if certificate.source != source_matrix:
            raise ValueError("Smith certificate is rebound to another source matrix")
        return _accept(
            "independent Smith transformation-certificate replay accepted "
            "matrix.normal_form.smith.certified.compute"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return _reject("malformed, unsupported, or mismatched checker request")


__all__ = [
    "ParsedMatrix",
    "ParsedSmithCertificate",
    "check_certified_smith_normal_form",
    "parse_matrix",
    "validate_certificate",
]
