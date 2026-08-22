"""Exact rational polytope operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian._exact import CanonicalRational


def convex_hull_volume(
    vertices: tuple[Sequence[Fraction], ...],
) -> CanonicalRational:
    """Return the exact rational volume of the convex hull of rational points.

    Accepts mathematical values — a non-empty tuple of rational coordinate
    tuples sharing one ambient dimension — and returns the canonical exact
    volume.  Degenerate inputs with fewer than ``dim + 1`` distinct points
    have exact volume zero.  Raises ``ValueError`` when the hull
    enumeration exceeds the combinatorial work bound.
    """

    from jacobian.math.polytope._operations import convex_hull_volume as _kernel

    normalized = tuple(tuple(Fraction(c) for c in vertex) for vertex in vertices)
    value, _dimension = _kernel(normalized)
    return value


__all__ = ["convex_hull_volume"]
