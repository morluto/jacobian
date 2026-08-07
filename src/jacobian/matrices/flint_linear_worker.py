"""Isolated Python-FLINT worker for exact rational linear-system candidates."""

from __future__ import annotations

import importlib
import re
import sys
from fractions import Fraction
from typing import Any

from jacobian.canonical import (
    CanonicalizationError,
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
)

FLINT_LINEAR_WORKER_PROTOCOL = "jacobian.flint-linear-worker/v1"
FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL = (
    "jacobian.flint-linear-inconsistency-worker/v1"
)
FLINT_LINEAR_INPUT_LIMIT = 1_000_000
MAX_LINEAR_DIMENSION = 32
MAX_RATIONAL_DIGITS = 256
_INTEGER = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_VARIABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


class FlintLinearWorkerError(RuntimeError):
    """One worker request could not produce usable evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _emit(
    payload: dict[str, object],
    *,
    protocol: str = FLINT_LINEAR_WORKER_PROTOCOL,
) -> None:
    sys.stdout.write(
        canonicalize_json({"protocol": protocol, **payload}).decode("utf-8") + "\n"
    )
    sys.stdout.flush()


def _read_request() -> dict[str, Any]:
    encoded = sys.stdin.buffer.read(FLINT_LINEAR_INPUT_LIMIT + 1)
    if len(encoded) > FLINT_LINEAR_INPUT_LIMIT:
        raise FlintLinearWorkerError("FLINT_LINEAR_INPUT_LIMIT_EXCEEDED")
    try:
        payload = loads_strict_json(encoded)
    except CanonicalizationError as exc:
        raise FlintLinearWorkerError("FLINT_LINEAR_INPUT_INVALID") from exc
    if not isinstance(payload, dict):
        raise FlintLinearWorkerError("FLINT_LINEAR_INPUT_INVALID")
    return payload


def _rational(value: Any) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    numerator = value["num"]
    denominator = value["den"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or _INTEGER.fullmatch(numerator) is None
        or _INTEGER.fullmatch(denominator) is None
        or len(numerator.lstrip("-")) > MAX_RATIONAL_DIGITS
        or len(denominator.lstrip("-")) > MAX_RATIONAL_DIGITS
    ):
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    try:
        result = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as exc:
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID") from exc
    if str(result.numerator) != numerator or str(result.denominator) != denominator:
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    return result


def _validate_system(
    request: dict[str, Any],
    *,
    protocol: str,
) -> tuple[list[list[Fraction]], list[Fraction]]:
    if set(request) != {"protocol", "system"} or request.get("protocol") != protocol:
        raise FlintLinearWorkerError("FLINT_LINEAR_INPUT_INVALID")
    system = request["system"]
    if not isinstance(system, dict) or set(system) != {
        "system_schema_version",
        "domain",
        "relation",
        "variables",
        "coefficients",
        "rhs",
    }:
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    if (
        system["system_schema_version"] != "1"
        or system["domain"] != "QQ"
        or system["relation"] != "AX_EQUALS_B"
    ):
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    variables = system["variables"]
    if (
        not isinstance(variables, list)
        or not 1 <= len(variables) <= MAX_LINEAR_DIMENSION
        or any(
            not isinstance(variable, str) or _VARIABLE.fullmatch(variable) is None
            for variable in variables
        )
        or len(set(variables)) != len(variables)
    ):
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    matrix = system["coefficients"]
    if not isinstance(matrix, dict) or set(matrix) != {
        "matrix_schema_version",
        "domain",
        "entries",
    }:
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    if matrix["matrix_schema_version"] != "1" or matrix["domain"] != "QQ":
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    entries = matrix["entries"]
    rhs = system["rhs"]
    if (
        not isinstance(entries, list)
        or not isinstance(rhs, list)
        or not 1 <= len(entries) <= MAX_LINEAR_DIMENSION
        or len(entries) != len(rhs)
        or any(
            not isinstance(row, list) or len(row) != len(variables) for row in entries
        )
    ):
        raise FlintLinearWorkerError("FLINT_LINEAR_SYSTEM_INVALID")
    return (
        [[_rational(value) for value in row] for row in entries],
        [_rational(value) for value in rhs],
    )


def _solve(
    coefficients: list[list[Fraction]],
    rhs: list[Fraction],
    flint: Any,
) -> list[Any] | None:
    augmented = flint.fmpq_mat(
        [
            [flint.fmpq(value.numerator, value.denominator) for value in row]
            + [flint.fmpq(bound.numerator, bound.denominator)]
            for row, bound in zip(coefficients, rhs, strict=True)
        ]
    )
    reduced, _ = augmented.rref()
    column_count = len(coefficients[0])
    values = [flint.fmpq(0) for _ in range(column_count)]
    for row_index in range(reduced.nrows()):
        pivot = next(
            (
                column
                for column in range(column_count)
                if reduced[row_index, column] != 0
            ),
            None,
        )
        if pivot is None:
            if reduced[row_index, column_count] != 0:
                return None
            continue
        values[pivot] = reduced[row_index, column_count]
    return values


def _run(request: dict[str, Any]) -> dict[str, object]:
    protocol = request.get("protocol")
    if not isinstance(protocol, str) or protocol not in {
        FLINT_LINEAR_WORKER_PROTOCOL,
        FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL,
    }:
        raise FlintLinearWorkerError("FLINT_LINEAR_INPUT_INVALID")
    coefficients, rhs = _validate_system(request, protocol=protocol)
    try:
        flint: Any = importlib.import_module("flint")
        if getattr(flint, "__version__", None) != "0.9.0":
            raise FlintLinearWorkerError("FLINT_LINEAR_VERSION_MISMATCH")
        if protocol == FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL:
            row_count = len(coefficients)
            column_count = len(coefficients[0])
            dual_coefficients = [
                [coefficients[row][column] for row in range(row_count)]
                for column in range(column_count)
            ]
            dual_coefficients.append(rhs)
            dual_rhs = [Fraction(0) for _ in range(column_count)] + [Fraction(1)]
            values = _solve(dual_coefficients, dual_rhs, flint)
        else:
            values = _solve(coefficients, rhs, flint)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as exc:
        raise FlintLinearWorkerError("FLINT_LINEAR_EXECUTION_FAILED") from exc
    if values is None:
        return {
            "status": (
                "NO_CERTIFICATE_PRODUCED"
                if protocol == FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL
                else "NO_SOLUTION_PRODUCED"
            ),
            "backend_version": "0.9.0",
        }
    if protocol == FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL:
        return {
            "status": "CERTIFICATE_PRODUCED",
            "backend_version": "0.9.0",
            "left_witness": [
                {
                    "num": format_canonical_integer(value.numerator),
                    "den": format_canonical_integer(value.denominator),
                }
                for value in values
            ],
            "rhs_pairing": {"num": "1", "den": "1"},
        }
    return {
        "status": "SOLUTION_PRODUCED",
        "backend_version": "0.9.0",
        "values": [
            {
                "num": format_canonical_integer(value.numerator),
                "den": format_canonical_integer(value.denominator),
            }
            for value in values
        ],
    }


def main() -> int:
    protocol = FLINT_LINEAR_WORKER_PROTOCOL
    try:
        request = _read_request()
        if request.get("protocol") == FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL:
            protocol = FLINT_LINEAR_INCONSISTENCY_WORKER_PROTOCOL
        result = _run(request)
    except FlintLinearWorkerError as exc:
        _emit({"error_code": exc.code}, protocol=protocol)
        return 2
    _emit(result, protocol=protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
