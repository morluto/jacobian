"""Spanned-line profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.spanned_line_profile._models import (
    SpannedLineEntry,
    SpannedLineProfileResult,
    _require_coordinate_distinctness,
)

__all__ = ["compute_spanned_line_profile", "verify_spanned_line_profile"]


def _admit_line_key_growth(configuration: PointConfiguration) -> None:
    points = configuration.points
    dimension = len(points[0].coordinates) if points else 0
    maximum_coordinate_digits = max(
        (
            max(len(coordinate.num.lstrip("-")), len(coordinate.den))
            for point in points
            for coordinate in point.coordinates
        ),
        default=1,
    )
    derived_digits = dimension * (2 * maximum_coordinate_digits + 2)
    if derived_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="geometry.spanned_line_profile.result_bound",
            message="spanned-line keys exceed the canonical rational digit bound",
        )


def compute_spanned_line_profile(
    configuration: PointConfiguration,
) -> SpannedLineProfileResult:
    """Return every distinct affine line spanned by unordered source pairs."""
    try:
        _require_coordinate_distinctness(configuration)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("configuration",), code=exc.type, message=exc.message()
        ) from exc
    _admit_line_key_growth(configuration)
    points = configuration.points
    n = len(points)

    point_coords = tuple(
        tuple(c.as_fraction() for c in point.coordinates) for point in points
    )

    line_to_pairs: dict[
        tuple[tuple[Fraction, ...], tuple[Fraction, ...]], list[tuple[int, int]]
    ] = {}

    for i, j in combinations(range(n), 2):
        ci = point_coords[i]
        cj = point_coords[j]
        if ci == cj:
            continue

        key = _canonical_line(ci, cj)
        if key is not None:
            line_to_pairs.setdefault(key, []).append((i, j))

    entries: list[SpannedLineEntry] = []
    for _key, pairs in line_to_pairs.items():
        all_indices: set[int] = set()
        for i, j in pairs:
            all_indices.add(i)
            all_indices.add(j)
        entries.append(
            SpannedLineEntry(
                source_pairs=tuple(pairs),
                point_count=len(all_indices),
            )
        )

    entries.sort(key=lambda e: e.source_pairs[0])

    return SpannedLineProfileResult(
        configuration=configuration,
        lines=tuple(entries),
        line_count=len(entries),
    )


def verify_spanned_line_profile(claim: SpannedLineProfileResult) -> bool:
    """Verify a serialized line profile against its retained configuration."""
    try:
        return compute_spanned_line_profile(claim.configuration) == claim
    except (OperationDomainValidationError, ValueError, RuntimeError):
        return False


def _canonical_line(
    ci: tuple[Fraction, ...],
    cj: tuple[Fraction, ...],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]] | None:
    """Return a canonical key for the line through ci and cj.

    The key is (direction, anchor_on_line) where direction is normalized
    so the first nonzero component is positive, and anchor is the projection
    of the origin onto the line (or ci if the projection is not rational-
    clean, we use ci itself since two pairs on the same line will share ci
    if they share a point, or we use the midpoint projected to canonical form).

    Actually, the simplest correct approach: for two pairs to span the same
    line, they must have the same direction AND one point from each pair must
    lie on the same line. We can check this by verifying that the vector from
    one pair's midpoint to the other pair's midpoint is parallel to the direction.

    But for a hash key, we need something simpler. Let's use:
    (normalized_direction, t0) where t0 is the parameter of the first nonzero
    coordinate's projection onto the line. For a line with direction d through
    point p, the parameter is p[k]/d[k] where k is the first nonzero direction index.
    """
    dim = len(ci)
    direction = tuple(cj[c] - ci[c] for c in range(dim))

    if all(d == 0 for d in direction):
        return None

    # Normalize by the first nonzero component, so reversing the source pair
    # produces exactly the same direction key with leading component one.
    first_nonzero_idx = next(i for i, d in enumerate(direction) if d != 0)
    first_nonzero = direction[first_nonzero_idx]
    norm_direction = tuple(d / first_nonzero for d in direction)

    # Anchor: the line is { ci + t * direction }. We need a canonical point.
    # Use the projection of the origin onto the line:
    # t0 = -dot(ci, direction) / dot(direction, direction)
    # anchor = ci + t0 * direction
    dot_ci_d = sum(ci[c] * direction[c] for c in range(dim))
    dot_d_d = sum(d * d for d in direction)
    if dot_d_d == 0:
        return None
    t0 = -dot_ci_d / dot_d_d
    anchor = tuple(ci[c] + t0 * direction[c] for c in range(dim))

    return (norm_direction, anchor)
