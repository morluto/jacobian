"""Standalone SymPy worker for admitted general rational gradients."""

from __future__ import annotations

import json
import sys
from typing import Any


def _polynomial(records: list[Any], symbols: tuple[Any, ...]) -> Any:
    from sympy import QQ, Poly, Rational

    coefficients: dict[tuple[int, ...], Any] = {}
    variable_count = len(symbols)
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


def _dump(polynomial: Any) -> list[list[Any]]:
    return [
        [*exponents, str(coefficient.p), str(coefficient.q)]
        for exponents, coefficient in polynomial.terms()
    ]


def _cancel(numerator: Any, denominator: Any) -> tuple[Any, Any]:
    from sympy import Poly

    if not numerator.is_zero:
        numerator_terms, denominator_terms = numerator.terms(), denominator.terms()
        common = tuple(
            min(
                min(exponents[axis] for exponents, _ in numerator_terms),
                min(exponents[axis] for exponents, _ in denominator_terms),
            )
            for axis in range(len(numerator.gens))
        )
        if any(common):

            def divide(value: Any) -> Any:
                return Poly.from_dict(
                    {
                        tuple(
                            exponent - shift
                            for exponent, shift in zip(exponents, common, strict=True)
                        ): coefficient
                        for exponents, coefficient in value.terms()
                    },
                    value.gens,
                    domain=value.domain,
                )

            numerator, denominator = divide(numerator), divide(denominator)
    numerator, denominator = numerator.cancel(denominator, include=True)
    leading = denominator.LC()
    return numerator.mul_ground(1 / leading), denominator.mul_ground(1 / leading)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import Symbol

    if set(payload) != {"variables", "numerator", "denominator"}:
        raise ValueError("malformed general-gradient request")
    variables = payload["variables"]
    if (
        not isinstance(variables, list)
        or not variables
        or any(not isinstance(name, str) or not name for name in variables)
        or not isinstance(payload["numerator"], list)
        or not isinstance(payload["denominator"], list)
    ):
        raise ValueError("malformed general-gradient request")
    generators = tuple(Symbol(name) for name in variables)
    numerator = _polynomial(payload["numerator"], generators)
    denominator = _polynomial(payload["denominator"], generators)
    if not numerator.gcd(denominator).is_one:
        return {"status": "noncanonical"}
    cancelled: list[dict[str, Any]] = []
    for axis in range(len(variables)):
        raw_numerator = numerator.diff(
            axis
        ) * denominator - numerator * denominator.diff(axis)
        raw_denominator = denominator * denominator
        reduced_numerator, reduced_denominator = _cancel(raw_numerator, raw_denominator)
        cancelled.append(
            {
                "numerator": _dump(reduced_numerator),
                "denominator": _dump(reduced_denominator),
            }
        )
    return {"status": "ok", "fractions": cancelled}


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed general-gradient request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
