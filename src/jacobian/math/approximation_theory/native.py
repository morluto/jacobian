"""Native domain functions over canonical approximation-theory values."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian._exact import CanonicalRational
from jacobian.math.approximation_theory._models import (
    LagrangeInterpolationRequest,
    RationalNodeSet,
)
from jacobian.math.approximation_theory._operations import (
    compute_lagrange_interpolation,
)
from jacobian.math.polynomials.values import RationalPolynomial

__all__ = ["lagrange_interpolate"]


def lagrange_interpolate(
    nodes: Sequence[CanonicalRational],
    values: Sequence[CanonicalRational],
) -> RationalPolynomial:
    """Exact polynomial of degree below ``len(nodes)`` through the samples.

    Accepts distinct increasing rational nodes and matching values as
    canonical domain values and returns the unique interpolant in the
    domain-owned :class:`RationalPolynomial` representation.
    """
    request = LagrangeInterpolationRequest(
        nodes=RationalNodeSet(nodes=tuple(nodes)),
        values=tuple(values),
    )
    return compute_lagrange_interpolation(request).polynomial
