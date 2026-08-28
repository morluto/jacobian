"""Exact native operations for hyperplane arrangements."""

from __future__ import annotations

import sympy

from jacobian.math.geometry.arrangements._models import (
    ChamberCountRequest,
    ChamberCountResult,
    CharacteristicPolynomialRequest,
    CharacteristicPolynomialResult,
    HyperplaneArrangementRequest,
    HyperplaneArrangementResult,
)


def compute_arrangement(
    request: HyperplaneArrangementRequest,
) -> HyperplaneArrangementResult:
    """Check if an arrangement is central (all hyperplanes pass through origin)."""
    is_central = all(
        hyperplane.constant.as_fraction() == 0 for hyperplane in request.hyperplanes
    )
    return HyperplaneArrangementResult(
        hyperplane_count=len(request.hyperplanes),
        ambient_dimension=request.ambient_dimension,
        is_central=is_central,
    )


def compute_characteristic_polynomial(
    request: CharacteristicPolynomialRequest,
) -> CharacteristicPolynomialResult:
    r"""Compute the characteristic polynomial of a generic central arrangement."""
    t = sympy.Symbol("t")
    n = request.ambient_dimension
    m = request.hyperplane_count
    inner = sum(
        (-1) ** k * sympy.binomial(m - 1, k) * t ** (n - 1 - k)
        for k in range(n)
    )
    coefficients = sympy.expand((t - 1) * inner).as_poly().all_coeffs()
    return CharacteristicPolynomialResult(
        coefficients=tuple(str(coefficient) for coefficient in reversed(coefficients)),
        degree=n,
    )


def compute_chamber_count(request: ChamberCountRequest) -> ChamberCountResult:
    r"""Count chambers of a generic central arrangement."""
    count = 2 * sum(
        sympy.binomial(request.hyperplane_count - 1, k)
        for k in range(request.ambient_dimension)
    )
    return ChamberCountResult(chamber_count=int(count))


__all__ = [
    "compute_arrangement",
    "compute_chamber_count",
    "compute_characteristic_polynomial",
]
