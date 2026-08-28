"""Native domain functions over canonical approximation-theory values."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian._exact import CanonicalRational
from jacobian.math.analysis.approximation._models import (
    RationalNodeSet,
)
from jacobian.math.analysis.approximation._operations import (
    _interpolate,
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
    node_set = RationalNodeSet(nodes=tuple(nodes))
    canonical_values = tuple(values)
    return _interpolate(node_set, canonical_values).polynomial
