"""Focused production services and payloads for polynomial operation tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from sympy import Poly, expand, symbols
from tests.support.catalog_build_options import CheckerAuthorityMode
from tests.support.services import (
    DomainTestServices,
    atomic_installation,
    open_domain_services,
)

from jacobian.polynomials import build_polynomial_operations
from jacobian.polynomials.resources import PolynomialContracts


@dataclass(frozen=True, slots=True)
class PolynomialTestServices(DomainTestServices):
    """The exact production graph owned by polynomial operation behavior."""

    polynomial: PolynomialContracts


@contextmanager
def open_polynomial_services(
    root: Path,
    *,
    checker_authority: CheckerAuthorityMode = CheckerAuthorityMode.NONE,
) -> Iterator[PolynomialTestServices]:
    """Install only generic verification and polynomial-map operations."""

    with open_domain_services(
        root,
        checker_authority=checker_authority,
    ) as services:
        with atomic_installation(services.core):
            adapters, polynomial = build_polynomial_operations(
                services.core.store,
                services.core.schemas,
                services.core.artifacts,
                services.verification,
                services.core.checkers,
                authorize_checker=services.installation.authorize_bundled_checkers,
            )
            for adapter in adapters:
                services.installation.register_operation(adapter)
        yield PolynomialTestServices(
            core=services.core,
            verification=services.verification,
            polytope=services.polytope,
            installation=services.installation,
            polynomial=polynomial,
        )


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
