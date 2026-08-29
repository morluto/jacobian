"""Domain-owned convex analysis operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import format_canonical_rational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.analysis.convex._models import (
    AffinePiece,
    MaxAffineEvalResult,
    MaxAffineFunction,
    MaxAffineSubdifferentialResult,
    RationalPoint,
)


def _admit_point(
    function: MaxAffineFunction,
    point: RationalPoint,
) -> None:
    dimension = len(function.pieces[0].coefficients)
    if len(point.coordinates) != dimension:
        raise OperationDomainValidationError(
            location=("point",),
            code="convex_analysis.point_dimension_mismatch",
            message="point dimension must match function dimension",
        )


def _evaluate_piece(piece: AffinePiece, point_coords: Any) -> Fraction:
    """Evaluate one affine piece at a point."""
    value = piece.intercept.as_fraction()
    for coeff, coord in zip(piece.coefficients, point_coords, strict=True):
        value += coeff.as_fraction() * coord.as_fraction()
    return value


def max_affine_evaluation(
    function: MaxAffineFunction,
    point: RationalPoint,
) -> MaxAffineEvalResult:
    """Evaluate f(x) = max_i { <a_i, x> + b_i } and identify active pieces."""
    _admit_point(function, point)
    point_coords = point.coordinates
    values = []
    active_pieces = []
    max_value = None

    for piece in function.pieces:
        v = _evaluate_piece(piece, point_coords)
        values.append((piece.piece_id, v))
        if max_value is None or v > max_value:
            max_value = v
            active_pieces = [piece.piece_id]
        elif v == max_value:
            active_pieces.append(piece.piece_id)

    all_values = tuple((pid, format_canonical_rational(v)) for pid, v in values)
    assert max_value is not None
    return MaxAffineEvalResult(
        value=format_canonical_rational(max_value),
        active_pieces=tuple(active_pieces),
        all_values=all_values,
    )


def max_affine_subdifferential(
    function: MaxAffineFunction,
    point: RationalPoint,
) -> MaxAffineSubdifferentialResult:
    """Compute the subdifferential at a point.

    The subdifferential of a max-affine function at x is the convex hull
    of the gradients of all active pieces. Here we return the gradients
    (coefficient vectors) of all active pieces.
    """
    _admit_point(function, point)
    point_coords = point.coordinates
    max_value = None
    active_gradients = []

    for piece in function.pieces:
        v = _evaluate_piece(piece, point_coords)
        if max_value is None or v > max_value:
            max_value = v
            active_gradients = [piece]
        elif v == max_value:
            active_gradients.append(piece)

    active_grads = tuple(piece.coefficients for piece in active_gradients)
    return MaxAffineSubdifferentialResult(active_gradients=active_grads)


__all__ = [
    "max_affine_evaluation",
    "max_affine_subdifferential",
]
