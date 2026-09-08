"""Standalone SymPy worker for forced denominator-derivative GCDs."""

from __future__ import annotations

import json
import sys
from typing import Any


def _polynomial(
    records: list[list[Any]], variable_count: int, symbols: tuple[Any, ...]
) -> Any:
    from sympy import QQ, Poly, Rational

    coefficients: dict[tuple[int, ...], Any] = {}
    for record in records:
        if not isinstance(record, list) or len(record) != variable_count + 2:
            raise ValueError("malformed polynomial record")
        exponents = tuple(record[:variable_count])
        numerator = record[-2]
        denominator = record[-1]
        if (
            any(type(exponent) is not int or exponent < 0 for exponent in exponents)
            or not isinstance(numerator, str)
            or not isinstance(denominator, str)
        ):
            raise ValueError("malformed polynomial record")
        coefficients[exponents] = Rational(int(numerator), int(denominator))
    return Poly.from_dict(coefficients, *symbols, domain=QQ)


def _factor_payload(factor: Any, variable_count: int) -> dict[str, Any]:
    terms = factor.terms()
    if not terms:
        return {
            "term_count": 0,
            "degrees": [0] * variable_count,
            "total_degree": 0,
            "minimum_exponents": [0] * variable_count,
        }
    exponents = tuple(monomial for monomial, _ in terms)
    return {
        "term_count": len(terms),
        "degrees": [
            max(monomial[index] for monomial in exponents)
            for index in range(variable_count)
        ],
        "total_degree": max(sum(monomial) for monomial in exponents),
        "minimum_exponents": [
            min(monomial[index] for monomial in exponents)
            for index in range(variable_count)
        ],
    }


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import symbols

    if set(payload) != {"variable_count", "terms"}:
        raise ValueError("malformed gcd request")
    variable_count = payload["variable_count"]
    records = payload["terms"]
    if (
        type(variable_count) is not int
        or variable_count <= 0
        or not isinstance(records, list)
    ):
        raise ValueError("malformed gcd request")
    generators = symbols(f"x0:{variable_count}")
    denominator = _polynomial(records, variable_count, generators)
    return {
        "factors": [
            _factor_payload(denominator.gcd(denominator.diff(axis)), variable_count)
            for axis in range(variable_count)
        ]
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed gcd request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
