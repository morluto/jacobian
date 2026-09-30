"""Catalog-publication checks moved out of the math lane.

This module boots the built-in catalog, which discovers and compiles every
owner-local ``._tools`` manifest. It lives under ``tests/catalog`` because
``tests/math`` must drive kernels directly and must not start the catalog;
see ``docs/reference/testing-strategy.md`` (test lane ownership).

Moved from ``tests/math/polynomials/test_quartic_cubic_resolvent.py``. The preamble below is carried over verbatim so the moved
tests resolve every name they use.
"""

import json
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.builtins import BUILTIN_TOOLS
from jacobian.math.polynomials._quartic_resolvent import (
    QuarticCubicResolventRequest,
)
from jacobian.math.polynomials._quartic_resolvent_tools import (
    QUARTIC_CUBIC_RESOLVENT_OPERATION,
)
from jacobian.math.polynomials.values import (
    MonicPolynomial,
    monic_polynomial_from_coefficients,
)


def _q(value: int) -> CanonicalRational:
    return CanonicalRational(num=value, den=1)


def _monic(coefficients: tuple[Fraction, ...]) -> MonicPolynomial:
    return monic_polynomial_from_coefficients(
        tuple(CanonicalRational.from_fraction(value) for value in coefficients),
        variable="t",
    )


def _mul(left: list[Fraction], right: list[Fraction]) -> list[Fraction]:
    product = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            product[i + j] += a * b
    return product


def _expand_roots(roots: tuple[Fraction, ...]) -> list[Fraction]:
    monic = [Fraction(1)]
    for root in roots:
        monic = _mul(monic, [-root, Fraction(1)])
    return monic


def _evaluate(poly: list[Fraction], value: Fraction) -> Fraction:
    total = Fraction(0)
    for coefficient in reversed(poly):
        total = total * value + coefficient
    return total


def _resultant(f: list[Fraction], g: list[Fraction]) -> Fraction:
    # Sylvester matrix determinant over QQ by exact Gaussian elimination.
    m = len(f) - 1
    n = len(g) - 1
    size = m + n
    rows: list[list[Fraction]] = []
    for shift in range(n):
        row = [Fraction(0)] * size
        row[shift : shift + m + 1] = list(f)
        rows.append(row)
    for shift in range(m):
        row = [Fraction(0)] * size
        row[shift : shift + n + 1] = list(g)
        rows.append(row)
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((r for r in range(column, size) if rows[r][column] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            rows[column], rows[pivot] = rows[pivot], rows[column]
            determinant = -determinant
        determinant *= rows[column][column]
        inverse = Fraction(1) / rows[column][column]
        for other in range(column + 1, size):
            factor = rows[other][column] * inverse
            if factor != 0:
                rows[other] = [
                    a - factor * b
                    for a, b in zip(rows[other], rows[column], strict=True)
                ]
    return determinant


def _discriminant(poly: list[Fraction]) -> Fraction:
    degree = len(poly) - 1
    derivative = [i * poly[i] for i in range(1, len(poly))]
    sign = -1 if degree * (degree - 1) // 2 % 2 else 1
    return sign * _resultant(poly, derivative) / poly[-1]


def test_operation_is_catalogued_with_executable_independent_example() -> None:
    assert QUARTIC_CUBIC_RESOLVENT_OPERATION.operation_id == (
        "polynomial.quartic.cubic_resolvent.compute"
    )
    assert QUARTIC_CUBIC_RESOLVENT_OPERATION in BUILTIN_TOOLS
    result = QUARTIC_CUBIC_RESOLVENT_OPERATION.run(
        QuarticCubicResolventRequest.model_validate_json(
            json.dumps(QUARTIC_CUBIC_RESOLVENT_OPERATION.examples[0].input)
        )
    )
    assert tuple(c.as_fraction() for c in result.resolvent.coefficients) == tuple(
        map(Fraction, (-1540, 404, -35, 1))
    )
