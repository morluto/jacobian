"""Isolated Python-FLINT worker for rational-linear candidates."""

from __future__ import annotations

import importlib
import sys
from fractions import Fraction
from typing import Any

from jacobian.canonical import (
    canonicalize_json,
    format_canonical_integer,
    loads_strict_json,
)

SOLUTION_PROTOCOL = "jacobian.rational-linear-solution-worker/v1"
INCONSISTENCY_PROTOCOL = "jacobian.rational-linear-inconsistency-worker/v1"


def _rational(value: object) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise ValueError("invalid rational")
    return Fraction(int(value["num"]), int(value["den"]))


def _solve(
    coefficients: list[list[Fraction]], rhs: list[Fraction], flint: Any
) -> list[Any] | None:
    augmented = flint.fmpq_mat(
        [
            [flint.fmpq(value.numerator, value.denominator) for value in row]
            + [flint.fmpq(bound.numerator, bound.denominator)]
            for row, bound in zip(coefficients, rhs, strict=True)
        ]
    )
    reduced, _ = augmented.rref()
    columns = len(coefficients[0])
    values = [flint.fmpq(0) for _ in range(columns)]
    for row in range(reduced.nrows()):
        pivot = next(
            (column for column in range(columns) if reduced[row, column] != 0), None
        )
        if pivot is None:
            if reduced[row, columns] != 0:
                return None
            continue
        values[pivot] = reduced[row, columns]
    return values


def _run(payload: dict[str, Any]) -> dict[str, object]:
    protocol = payload.get("protocol")
    if protocol not in {SOLUTION_PROTOCOL, INCONSISTENCY_PROTOCOL}:
        raise ValueError("invalid protocol")
    system = payload["system"]
    matrix = system["coefficients"]["entries"]
    rhs = system["rhs"]
    coefficients = [[_rational(value) for value in row] for row in matrix]
    bounds = [_rational(value) for value in rhs]
    flint: Any = importlib.import_module("flint")
    if getattr(flint, "__version__", None) != "0.9.0":
        raise ValueError("unsupported Python-FLINT version")
    if protocol == INCONSISTENCY_PROTOCOL:
        row_count = len(coefficients)
        column_count = len(coefficients[0])
        dual = [
            [coefficients[row][column] for row in range(row_count)]
            for column in range(column_count)
        ]
        dual.append(bounds)
        values = _solve(dual, [Fraction(0)] * column_count + [Fraction(1)], flint)
        if values is None:
            return {"status": "NO_CERTIFICATE_PRODUCED"}
        return {
            "status": "CERTIFICATE_PRODUCED",
            "left_witness": [
                {
                    "num": format_canonical_integer(value.numerator),
                    "den": format_canonical_integer(value.denominator),
                }
                for value in values
            ],
            "rhs_pairing": {"num": "1", "den": "1"},
        }
    values = _solve(coefficients, bounds, flint)
    if values is None:
        return {"status": "NO_SOLUTION_PRODUCED"}
    return {
        "status": "SOLUTION_PRODUCED",
        "values": [
            {
                "num": format_canonical_integer(value.numerator),
                "den": format_canonical_integer(value.denominator),
            }
            for value in values
        ],
    }


def main() -> int:
    try:
        payload = loads_strict_json(sys.stdin.buffer.read())
        if not isinstance(payload, dict):
            raise ValueError("invalid request")
        result = _run(payload)
        protocol = payload["protocol"]
        sys.stdout.buffer.write(
            canonicalize_json({"protocol": protocol, **result}) + b"\n"
        )
        return 0
    except Exception as exc:  # pragma: no cover - process boundary
        protocol = SOLUTION_PROTOCOL
        sys.stdout.buffer.write(
            canonicalize_json(
                {"protocol": protocol, "status": "ERROR", "error": type(exc).__name__}
            )
            + b"\n"
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
