"""Domain functions for hyperplane arrangement operations."""

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
    is_central = True
    for hp in request.hyperplanes:
        if hp.constant.as_fraction() != 0:
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
    r"""Compute the characteristic polynomial of a generic central arrangement.

    For a generic central arrangement of m hyperplanes in R^n, the
    characteristic polynomial is::

        chi(t) = (t - 1) * sum_{k=0}^{n-1} (-1)^k * C(m-1, k) * t^{n-1-k}

    This is always monic of degree n.  When m <= n it coincides with the
    general-position formula ``sum_{k=0}^{m} (-1)^k C(m,k) t^{n-k}``.
    """
    t = sympy.Symbol("t")
    n = request.ambient_dimension
    m = request.hyperplane_count
    inner = sum(
        (-1) ** k * sympy.binomial(m - 1, k) * t ** (n - 1 - k) for k in range(n)
    )
    poly = sympy.expand((t - 1) * inner)
    coeffs = poly.as_poly().all_coeffs()
    coeffs_str = tuple(str(c) for c in reversed(coeffs))
    return CharacteristicPolynomialResult(
        coefficients=coeffs_str,
        degree=n,
    )


def compute_chamber_count(request: ChamberCountRequest) -> ChamberCountResult:
    r"""Count the number of chambers of a generic central arrangement.

    For a generic central (linear) arrangement of m hyperplanes in R^n, the
    number of chambers (regions) is::

        r = 2 * sum_{k=0}^{n-1} C(m-1, k)

    This is consistent with the characteristic polynomial via Zaslavsky's
    theorem: ``r = (-1)^n * chi(-1)``.
    """
    n = request.ambient_dimension
    m = request.hyperplane_count
    count = 2 * sum(sympy.binomial(m - 1, k) for k in range(n))
    return ChamberCountResult(
        chamber_count=int(count),
    )
