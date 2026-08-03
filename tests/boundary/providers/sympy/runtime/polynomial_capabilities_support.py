"""Payload builders shared by the SymPy polynomial capability tests."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from sympy import Poly, expand, symbols


def wire_fraction(value: Fraction | int) -> dict[str, str]:
    rational = Fraction(value)
    return {"num": str(rational.numerator), "den": str(rational.denominator)}


def poly_payload(poly: Poly) -> dict[str, Any]:
    return {
        "terms": [
            {
                "coefficient": wire_fraction(Fraction(coefficient)),
                "exponents": list(exponents),
            }
            for exponents, coefficient in poly.terms()
        ]
    }


def jacobian_counterexample_map() -> dict[str, Any]:
    x, y, z = symbols("x y z")
    coordinates = (
        (1 + x * y) ** 3 * z + y**2 * (1 + x * y) * (4 + 3 * x * y),
        y + 3 * x * (1 + x * y) ** 2 * z + 3 * x * y**2 * (4 + 3 * x * y),
        2 * x - 3 * x**2 * y - x**3 * z,
    )
    return {
        "map_schema_version": "1",
        "domain": "QQ",
        "variables": ["x", "y", "z"],
        "coordinates": [
            poly_payload(Poly(expand(coordinate), x, y, z, domain="QQ"))
            for coordinate in coordinates
        ],
    }


def point(*values: Fraction | int) -> list[dict[str, str]]:
    return [wire_fraction(value) for value in values]


def identity_input(*, right_coefficient: Fraction | int = 2) -> dict[str, Any]:
    return {
        "variables": ["x", "y"],
        "left": {
            "terms": [
                {"coefficient": wire_fraction(2), "exponents": [2, 0]},
                {"coefficient": wire_fraction(-1), "exponents": [0, 1]},
            ]
        },
        "right": {
            "terms": [
                {
                    "coefficient": wire_fraction(right_coefficient),
                    "exponents": [2, 0],
                },
                {"coefficient": wire_fraction(-1), "exponents": [0, 1]},
            ]
        },
    }


def rational_function_identity_input(*, equal: bool = True) -> dict[str, Any]:
    x = symbols("x")
    return {
        "variables": ["x"],
        "left": {
            "numerator": poly_payload(Poly(x**2 - 1, x, domain="QQ")),
            "denominator": poly_payload(Poly(x - 1, x, domain="QQ")),
        },
        "right": {
            "numerator": poly_payload(Poly(x + (1 if equal else 2), x, domain="QQ")),
            "denominator": poly_payload(Poly(1, x, domain="QQ")),
        },
    }
