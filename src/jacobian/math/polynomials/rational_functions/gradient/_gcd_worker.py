"""Standalone SymPy worker for rational-gradient GCDs and normalization."""

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


def _records(polynomial: Any, variable_count: int) -> list[list[Any]]:
    if polynomial.is_zero:
        return []
    return [
        [
            *exponents,
            str(coefficient.p),
            str(coefficient.q),
        ]
        for exponents, coefficient in polynomial.terms()
    ]


def _factor_payload(factor: Any, variable_count: int) -> dict[str, Any]:
    terms = factor.terms()
    if not terms:
        return {
            "term_count": 0,
            "degrees": [0] * variable_count,
            "total_degree": 0,
            "minimum_exponents": [0] * variable_count,
            "terms": [],
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
        "terms": _records(factor, variable_count),
    }


def _remove_common_monomial(
    numerator: Any, denominator: Any, variable_count: int
) -> Any:
    from sympy import Poly

    if numerator.is_zero:
        return numerator, denominator
    numerator_terms, denominator_terms = numerator.terms(), denominator.terms()
    common = tuple(
        min(
            min(exponents[axis] for exponents, _ in numerator_terms),
            min(exponents[axis] for exponents, _ in denominator_terms),
        )
        for axis in range(variable_count)
    )
    if not any(common):
        return numerator, denominator

    def divide(value: Any) -> Any:
        return Poly.from_dict(
            {
                tuple(
                    e - c for e, c in zip(exponents, common, strict=True)
                ): coefficient
                for exponents, coefficient in value.terms()
            },
            value.gens,
            domain=value.domain,
        )

    return divide(numerator), divide(denominator)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import symbols

    task = payload.get("task")
    variable_count = payload.get("variable_count")
    if type(variable_count) is not int or variable_count <= 0:
        raise ValueError("malformed kernel request")
    generators = symbols(f"x0:{variable_count}")
    if task == "derivative_gcds":
        if set(payload) != {"task", "variable_count", "terms"}:
            raise ValueError("malformed kernel request")
        records = payload["terms"]
        if not isinstance(records, list):
            raise ValueError("malformed kernel request")
        denominator = _polynomial(records, variable_count, generators)
        return {
            "factors": [
                _factor_payload(denominator.gcd(denominator.diff(axis)), variable_count)
                for axis in range(variable_count)
            ]
        }
    if task == "coprime":
        if set(payload) != {"task", "variable_count", "numerator", "denominator"}:
            raise ValueError("malformed kernel request")
        numerator_records = payload["numerator"]
        denominator_records = payload["denominator"]
        if not isinstance(numerator_records, list) or not isinstance(
            denominator_records, list
        ):
            raise ValueError("malformed kernel request")
        numerator = _polynomial(numerator_records, variable_count, generators)
        denominator = _polynomial(denominator_records, variable_count, generators)
        return {"coprime": bool(numerator.gcd(denominator).is_one)}
    if task == "normalize":
        if set(payload) != {
            "task",
            "variable_count",
            "numerator",
            "denominator",
            "factor",
        }:
            raise ValueError("malformed kernel request")
        numerator_records = payload["numerator"]
        denominator_records = payload["denominator"]
        factor_records = payload["factor"]
        if (
            not isinstance(numerator_records, list)
            or not isinstance(denominator_records, list)
            or not isinstance(factor_records, list)
        ):
            raise ValueError("malformed kernel request")
        numerator = _polynomial(numerator_records, variable_count, generators)
        denominator = _polynomial(denominator_records, variable_count, generators)
        if factor_records:
            factor = _polynomial(factor_records, variable_count, generators)
            if not factor.is_zero and not factor.is_one:
                numerator = numerator.exquo(factor)
                denominator = denominator.exquo(factor)
        numerator, denominator = _remove_common_monomial(
            numerator, denominator, variable_count
        )
        if numerator.is_zero:
            one = [0] * variable_count + ["1", "1"]
            return {
                "numerator": [],
                "denominator": [one],
            }
        numerator, denominator = numerator.cancel(denominator, include=True)
        leading = denominator.LC()
        numerator = numerator.mul_ground(1 / leading)
        denominator = denominator.mul_ground(1 / leading)
        return {
            "numerator": _records(numerator, variable_count),
            "denominator": _records(denominator, variable_count),
        }
    raise ValueError("malformed kernel request")


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed kernel request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
