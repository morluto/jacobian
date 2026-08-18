"""Domain functions for hyperplane arrangement operations."""

from __future__ import annotations

import sympy

from jacobian.math.hyperplane_arrangements._models import (
    ChamberCountRequest,
    ChamberCountResult,
    CharacteristicPolynomialRequest,
    CharacteristicPolynomialResult,
    HyperplaneArrangementRequest,
    HyperplaneArrangementResult,
)


def compute_arrangement(request: HyperplaneArrangementRequest) -> HyperplaneArrangementResult:
    """Check if an arrangement is central (all hyperplanes pass through origin)."""
    is_central = True
    for hp in request.hyperplanes:
        if sympy.sympify(hp.constant) != 0:
            is_central = False
            break
    return HyperplaneArrangementResult(
        hyperplane_count=len(request.hyperplanes),
        ambient_dimension=request.ambient_dimension,
        is_central=is_central,
    )


def compute_characteristic_polynomial(
    request: CharacteristicPolynomialRequest,
) -> CharacteristicPolynomialResult:
    """Compute the characteristic polynomial of a generic central arrangement.

    For a generic arrangement of m hyperplanes in R^n, the characteristic
    polynomial is chi(t) = sum_{k=0}^{n} (-1)^k * C(m, k) * t^(n-k) * (m-k)^(n-k).
    For generic arrangements, this simplifies to the Zaslavsky formula.
    """
    t = sympy.Symbol("t")
    n = request.ambient_dimension
    m = request.hyperplane_count
    poly = sum(
        (-1) ** k * sympy.binomial(m, k) * t ** (n - k) * (m - k) ** (n - k)
        for k in range(min(n, m) + 1)
    )
    poly = sympy.expand(poly)
    coeffs = poly.as_poly().all_coeffs()
    coeffs_str = tuple(str(c) for c in reversed(coeffs))
    return CharacteristicPolynomialResult(
        coefficients=coeffs_str,
        degree=n,
    )


def compute_chamber_count(request: ChamberCountRequest) -> ChamberCountResult:
    """Count the number of chambers of a generic central arrangement.

    For a generic central arrangement of m hyperplanes in R^n, the number
    of chambers is sum_{k=0}^{n} C(m, k).
    """
    n = request.ambient_dimension
    m = request.hyperplane_count
    count = sum(sympy.binomial(m, k) for k in range(min(n, m) + 1))
    return ChamberCountResult(
        chamber_count=int(count),
    )
