"""Standalone SymPy worker for admitted rational-gradient quotient rule."""

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
        [
            *exponents,
            str(coefficient.p),
            str(coefficient.q),
        ]
        for exponents, coefficient in polynomial.terms()
    ]


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    from sympy import symbols

    if set(payload) != {"variable_count", "axis", "numerator", "denominator"}:
        raise ValueError("malformed gradient kernel request")
    variable_count = payload["variable_count"]
    axis = payload["axis"]
    if (
        type(variable_count) is not int
        or variable_count < 1
        or type(axis) is not int
        or not 0 <= axis < variable_count
        or not isinstance(payload["numerator"], list)
        or not isinstance(payload["denominator"], list)
    ):
        raise ValueError("malformed gradient kernel request")
    generators = symbols(f"x0:{variable_count}")
    numerator = _polynomial(payload["numerator"], generators)
    denominator = _polynomial(payload["denominator"], generators)
    generator = generators[axis]
    numerator = numerator.diff(generator) * denominator - numerator * denominator.diff(
        generator
    )
    if numerator.is_zero:
        from sympy import QQ, Poly

        one = Poly(1, *generators, domain=QQ)
        return {
            "status": "ok",
            "numerator": [],
            "denominator": _dump(one),
        }
    denominator = denominator * denominator
    if not numerator.is_zero:
        numerator_terms, denominator_terms = numerator.terms(), denominator.terms()
        common = tuple(
            min(
                min(exponents[index] for exponents, _ in numerator_terms),
                min(exponents[index] for exponents, _ in denominator_terms),
            )
            for index in range(variable_count)
        )
        if any(common):
            from sympy import Poly

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
    numerator = numerator.mul_ground(1 / leading)
    denominator = denominator.mul_ground(1 / leading)
    return {
        "status": "ok",
        "numerator": _dump(numerator),
        "denominator": _dump(denominator),
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("malformed gradient kernel request")
        response = _run(payload)
    except Exception:
        return 1
    sys.stdout.buffer.write(
        json.dumps(response, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
