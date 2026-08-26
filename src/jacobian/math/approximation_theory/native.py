"""Native domain functions over canonical approximation-theory values."""

from __future__ import annotations

from collections.abc import Sequence

from jacobian._exact import CanonicalRational
from jacobian.math.approximation_theory._models import (
    RationalNodeSet,
    admit_interpolation_values,
)
from jacobian.math.approximation_theory._operations import (
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
    admit_interpolation_values(node_set, canonical_values)
    return _interpolate(node_set, canonical_values).polynomial
