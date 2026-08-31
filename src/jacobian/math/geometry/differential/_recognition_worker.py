"""Standalone SymPy worker for admitted rational-function GCDs."""

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


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import symbols

    if set(payload) != {"candidates"} or not isinstance(payload["candidates"], list):
        raise ValueError("malformed recognition request")
    recognized = 0
    for candidate in payload["candidates"]:
        if not isinstance(candidate, dict) or set(candidate) != {
            "owner",
            "component",
            "variable_count",
            "numerator",
            "denominator",
        }:
            raise ValueError("malformed recognition candidate")
        owner = candidate["owner"]
        component = candidate["component"]
        variable_count = candidate["variable_count"]
        if (
            owner not in ("vector_field", "tensor")
            or type(component) is not int
            or component < 0
            or type(variable_count) is not int
            or variable_count <= 0
            or not isinstance(candidate["numerator"], list)
            or not isinstance(candidate["denominator"], list)
        ):
            raise ValueError("malformed recognition candidate")
        generators = symbols(f"x0:{variable_count}")
        numerator = _polynomial(candidate["numerator"], variable_count, generators)
        denominator = _polynomial(candidate["denominator"], variable_count, generators)
        if not numerator.gcd(denominator).is_one:
            return {
                "status": "NOT_COPRIME",
                "recognized_candidates": recognized + 1,
                "owner": owner,
                "component": component,
            }
        recognized += 1
    return {"status": "CANONICAL", "recognized_candidates": recognized}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed recognition request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
