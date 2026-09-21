"""Domain-owned convex analysis operations."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.analysis.convex._models import (
    MAX_CONVEX_OUTPUT_DIGITS,
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


def _integer_digits(value: int) -> int:
    """Count canonical decimal digits without relying on ``str(int)``."""
    return len(format_canonical_integer(abs(value)))


def _rational_height(value: CanonicalRational) -> tuple[int, int]:
    """Return numerator and denominator digit bounds for one source scalar."""
    return _integer_digits(value.num), _integer_digits(value.den)


def _product_height(
    left: tuple[int, int], right: tuple[int, int], *, left_zero: bool, right_zero: bool
) -> tuple[int, int]:
    """Bound a rational product before reduction, using only source heights."""
    if left_zero or right_zero:
        return 1, 1
    return left[0] + right[0], left[1] + right[1]


def _sum_height(left: tuple[int, int], right: tuple[int, int]) -> tuple[int, int]:
    """Bound one exact rational addition before reduction or cancellation."""
    return (
        max(left[0] + right[1], right[0] + left[1]) + 1,
        left[1] + right[1],
    )


def _admit_evaluation_height(
    function: MaxAffineFunction,
    point: RationalPoint,
    *,
    include_piece_values: bool,
) -> None:
    """Admit exact affine arithmetic and retained result digits once.

    This is a structural estimate: multiplication and addition are priced from
    component heights without evaluating the expression a second time.
    Reduction and cancellation can only lower the bounds.
    """
    source_digits = sum(
        _rational_height(value)[0] + _rational_height(value)[1]
        for piece in function.pieces
        for value in (*piece.coefficients, piece.intercept)
    ) + sum(
        _rational_height(value)[0] + _rational_height(value)[1]
        for value in point.coordinates
    )
    output_digits = source_digits + sum(
        len(piece.piece_id) for piece in function.pieces
    )
    for piece in function.pieces:
        current = _rational_height(piece.intercept)
        has_nonzero_term = piece.intercept.num != 0
        for coefficient, coordinate in zip(
            piece.coefficients, point.coordinates, strict=True
        ):
            term = _product_height(
                _rational_height(coefficient),
                _rational_height(coordinate),
                left_zero=coefficient.num == 0,
                right_zero=coordinate.num == 0,
            )
            if coefficient.num == 0 or coordinate.num == 0:
                continue
            current = term if not has_nonzero_term else _sum_height(current, term)
            has_nonzero_term = True
            if max(current) > MAX_CANONICAL_RATIONAL_DIGITS:
                raise OperationResourceAdmissionError(
                    location=("function", "pieces"),
                    code="convex_analysis.derived_height_exceeded",
                    message=(
                        "max-affine exact arithmetic exceeds the canonical "
                        f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit scalar bound: "
                        f"intermediate numerator={current[0]}, denominator={current[1]}"
                    ),
                )
        if include_piece_values:
            output_digits += sum(current)
    # Evaluation returns one value per piece and the selected maximum while
    # retaining the source function and point.
    if include_piece_values:
        output_digits += MAX_CANONICAL_RATIONAL_DIGITS
    else:
        output_digits += sum(
            _integer_digits(value.num) + _integer_digits(value.den)
            for piece in function.pieces
            for value in piece.coefficients
        )
    if output_digits > MAX_CONVEX_OUTPUT_DIGITS:
        raise OperationResourceAdmissionError(
            location=("function", "pieces"),
            code="convex_analysis.output_height_exceeded",
            message=(
                "max-affine retained source and result digits exceed the exact "
                f"output budget of {MAX_CONVEX_OUTPUT_DIGITS:,} digits"
            ),
        )


def max_affine_evaluation(
    function: MaxAffineFunction,
    point: RationalPoint,
) -> MaxAffineEvalResult:
    """Evaluate f(x) = max_i { <a_i, x> + b_i } and identify active pieces."""
    _admit_point(function, point)
    _admit_evaluation_height(function, point, include_piece_values=True)
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

    all_values = tuple((pid, CanonicalRational.from_fraction(v)) for pid, v in values)
    if max_value is None:
        raise RuntimeError("admitted max-affine function has no pieces")
    return MaxAffineEvalResult(
        function=function,
        point=point,
        value=CanonicalRational.from_fraction(max_value),
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
    _admit_evaluation_height(function, point, include_piece_values=False)
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
    return MaxAffineSubdifferentialResult(
        function=function,
        point=point,
        active_gradients=active_grads,
    )


def verify_max_affine_evaluation(claim: MaxAffineEvalResult) -> bool:
    """Verify value, active-piece IDs, and per-piece values against the source."""

    if not isinstance(claim, MaxAffineEvalResult):
        return False
    try:
        return max_affine_evaluation(claim.function, claim.point) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


def verify_max_affine_subdifferential(
    claim: MaxAffineSubdifferentialResult,
) -> bool:
    """Verify active gradients against the retained function and point."""

    if not isinstance(claim, MaxAffineSubdifferentialResult):
        return False
    try:
        return max_affine_subdifferential(claim.function, claim.point) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False


__all__ = [
    "max_affine_evaluation",
    "max_affine_subdifferential",
    "verify_max_affine_evaluation",
    "verify_max_affine_subdifferential",
]
