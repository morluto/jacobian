"""Public native constructors over finite-geometry canonical values."""

from __future__ import annotations

from jacobian.math.finite_geometry.values import (
    PrimeFieldVectorSpace,
    ProjectivePoint,
    _validate_vector,
)


def projective_point(
    space: PrimeFieldVectorSpace, vector: tuple[int, ...]
) -> ProjectivePoint:
    """Canonicalize one nonzero finite-field vector into its point.

    The vector must hold canonical field residues of ``space``; scaling its
    first nonzero entry to one returns the canonical projective
    representative bound to ``space``.
    """

    _validate_vector(vector, space)
    scale = next((value for value in vector if value != 0), None)
    if scale is None:
        raise ValueError("zero vector has no projective point")
    inverse = pow(scale, -1, space.field_order)
    return ProjectivePoint(
        space=space,
        coordinates=tuple(value * inverse % space.field_order for value in vector),
    )


__all__ = ["projective_point"]
