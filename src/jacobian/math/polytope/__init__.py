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
    have exact volume zero.

    Applies the same conservative admission as :class:`PolytopeVolumeRequest`
    (dimension, vertex count, coordinate size, and triangulation-aware
    result growth) before invoking the kernel, so every accepted native
    input has a representable exact volume.  Raises ``ValueError`` when an
    input leaves that admitted domain or the hull enumeration exceeds the
    combinatorial work bound.
    """

    if not vertices:
        raise ValueError("`vertices` must be non-empty")
    dim = len(vertices[0])
    if any(len(vertex) != dim for vertex in vertices):
        raise ValueError("all vertices must share one dimension")

    from jacobian.canonical import format_canonical_integer
    from jacobian.math.polytope._models import (
        COORDINATE_DIGITS,
        MAX_DIMENSION,
        MAX_VERTICES,
        require_volume_components_within_result_bound,
    )

    if not 1 <= dim <= MAX_DIMENSION:
        raise ValueError(
            f"ambient dimension {dim} exceeds the {MAX_DIMENSION}-dimension bound"
        )
    if len(vertices) > MAX_VERTICES:
        raise ValueError(f"`vertices` exceeds the {MAX_VERTICES}-vertex bound")
    normalized = tuple(tuple(Fraction(c) for c in vertex) for vertex in vertices)
    for vertex in normalized:
        for coord in vertex:
            num_digits = len(format_canonical_integer(abs(coord.numerator)))
            den_digits = len(format_canonical_integer(coord.denominator))
            if max(num_digits, den_digits) > COORDINATE_DIGITS:
                raise ValueError(
                    f"vertex coordinate exceeds the {COORDINATE_DIGITS}-digit bound"
                )
    require_volume_components_within_result_bound(normalized, dim)

    from jacobian.math.polytope._operations import convex_hull_volume as _kernel

    value, _dimension = _kernel(normalized)
    return value


__all__ = ["convex_hull_volume"]
