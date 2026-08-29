"""Exact native operations for hyperplane arrangements."""

from __future__ import annotations

import sympy

from jacobian.math.geometry.arrangements._models import (
    ChamberCountResult,
    CharacteristicPolynomialResult,
    HyperplaneArrangementResult,
    RationalHyperplane,
)


def arrangement(
    ambient_dimension: int,
    hyperplanes: tuple[RationalHyperplane, ...],
) -> HyperplaneArrangementResult:
    """Check if an arrangement is central (all hyperplanes pass through origin)."""
    is_central = all(
        hyperplane.constant.as_fraction() == 0 for hyperplane in hyperplanes
    )
    return HyperplaneArrangementResult(
        hyperplane_count=len(hyperplanes),
        ambient_dimension=ambient_dimension,
        is_central=is_central,
    )


def characteristic_polynomial(
    ambient_dimension: int,
    hyperplane_count: int,
) -> CharacteristicPolynomialResult:
    r"""Compute the characteristic polynomial of a generic central arrangement."""
    t = sympy.Symbol("t")
    n = ambient_dimension
    m = hyperplane_count
    inner = sum(
        (-1) ** k * sympy.binomial(m - 1, k) * t ** (n - 1 - k) for k in range(n)
    )
    coefficients = sympy.expand((t - 1) * inner).as_poly().all_coeffs()
    return CharacteristicPolynomialResult(
        coefficients=tuple(str(coefficient) for coefficient in reversed(coefficients)),
        degree=n,
    )


def chamber_count(ambient_dimension: int, hyperplane_count: int) -> ChamberCountResult:
    r"""Count chambers of a generic central arrangement."""
    count = 2 * sum(
        sympy.binomial(hyperplane_count - 1, k) for k in range(ambient_dimension)
    )
    return ChamberCountResult(chamber_count=int(count))


__all__ = [
    "arrangement",
    "chamber_count",
    "characteristic_polynomial",
]
