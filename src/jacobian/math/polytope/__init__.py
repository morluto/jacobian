"""Exact rational polytope operations."""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.polytope._models import (
    PolytopeSupportResult,
    RationalCoordinateSpace,
    RationalCovector,
    RationalExposedFace,
    RationalPolytopeVertex,
    RationalVPolytope,
    require_support_components_within_envelope,
)


def polytope_support(
    polytope: RationalVPolytope,
    covector: RationalCovector,
) -> PolytopeSupportResult:
    """Return the exact support value and complete exposed vertex face.

    Both arguments are canonical values from this domain. The V-polytope
    retains an ordered labelled coordinate axis and its complete irredundant
    full-dimensional vertex family, so the covector cannot be paired against
    an unrelated coordinate order. The returned source-bound result replays
    the maximum across every retained vertex.

    Applies the operation's smaller construction envelope before invoking
    the kernel: every vertex coordinate and covector component must carry at
    most 150 digits per reduced numerator or denominator (canonical values
    themselves admit up to the global canonical limit), and the two spaces
    must be identical. Raises ``ValueError`` outside that admitted domain.
    """

    require_support_components_within_envelope(polytope, covector)

    from jacobian.math.polytope._operations import polytope_support as _kernel

    return _kernel(polytope, covector)


def convex_hull_volume(
    vertices: RationalVPolytope | tuple[Sequence[Fraction], ...],
) -> CanonicalRational:
    """Return the exact rational volume of the convex hull of rational points.

    Accepts mathematical values: either the domain's canonical labelled
    :class:`RationalVPolytope` (for example the ``polytope`` of a
    ``polytope_support`` result, whose ordered coordinate axis fixes each
    vertex's component order) or a non-empty tuple of rational coordinate
    tuples sharing one ambient dimension.  Returns the canonical exact
    volume.  Degenerate inputs with fewer than ``dim + 1`` distinct points
    have exact volume zero.

    Applies the same conservative admission as :class:`PolytopeVolumeRequest`
    (dimension, vertex count, coordinate size, and triangulation-aware
    result growth) before invoking the kernel, so every accepted native
    input has a representable exact volume.  Raises ``ValueError`` when an
    input leaves that admitted domain or the hull enumeration exceeds the
    combinatorial work bound.
    """

    from jacobian.canonical import format_canonical_integer
    from jacobian.math.polytope._models import (
        COORDINATE_DIGITS,
        MAX_DIMENSION,
        MAX_VERTICES,
        _canonical_v_polytope_vertices,
        require_volume_components_within_result_bound,
    )

    if isinstance(vertices, RationalVPolytope):
        normalized = tuple(
            tuple(Fraction(*c.as_integer_ratio()) for c in vertex.coordinates)
            for vertex in _canonical_v_polytope_vertices(vertices)
        )
    else:
        if not vertices:
            raise ValueError("`vertices` must be non-empty")
        if any(len(vertex) != len(vertices[0]) for vertex in vertices):
            raise ValueError("all vertices must share one dimension")
        normalized = tuple(tuple(Fraction(c) for c in vertex) for vertex in vertices)

    dim = len(normalized[0])
    if not 1 <= dim <= MAX_DIMENSION:
        raise ValueError(
            f"ambient dimension {dim} exceeds the {MAX_DIMENSION}-dimension bound"
        )
    if len(normalized) > MAX_VERTICES:
        raise ValueError(f"`vertices` exceeds the {MAX_VERTICES}-vertex bound")
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


__all__ = [
    "PolytopeSupportResult",
    "RationalCoordinateSpace",
    "RationalCovector",
    "RationalExposedFace",
    "RationalPolytopeVertex",
    "RationalVPolytope",
    "convex_hull_volume",
    "polytope_support",
]
