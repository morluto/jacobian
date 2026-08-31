"""Triangle area profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import CanonicalLimits, format_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.triangle_area_profile._models import (
    TriangleAreaEntry,
    TriangleAreaProfileResult,
    _require_distinct_coordinates,
)

__all__ = ["compute_triangle_area_profile"]


def _admit_triangle_area_result(configuration: PointConfiguration) -> None:
    """Reject configurations whose complete profile cannot fit the wire limit."""
    points = configuration.points
    triangle_count = len(points) * (len(points) - 1) * (len(points) - 2) // 6
    source_bytes = sum(
        len(point.label)
        + sum(len(coord.num) + len(coord.den) + 16 for coord in point.coordinates)
        + 32
        for point in points
    )
    if triangle_count == 0:
        if source_bytes + 256 > CanonicalLimits().max_output_bytes:
            raise OperationDomainValidationError(
                location=("configuration",),
                code="geometry.triangle_area_result_bytes",
                message="triangle area profile exceeds the canonical output-byte limit",
            )
        return
    coordinate_widths = sorted(
        (
            max(len(coord.num.lstrip("-")), len(coord.den))
            for point in points
            for coord in point.coordinates
        ),
        reverse=True,
    )
    # A cross-product term can retain denominator factors from all six
    # coordinates in a triple; subtraction and the factor 1/2 add carry
    # digits. Reserve that complete factor product before enumeration.
    derived_digits = sum(coordinate_widths[:6], 0) + 2
    if derived_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="geometry.triangle_area_result_bound",
            message=(
                "a derived triangle area exceeds the canonical rational "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )
    entry_bytes = 96 + 2 * derived_digits
    class_bytes = 128 + 2 * derived_digits
    estimated_bytes = source_bytes + 256 + triangle_count * (entry_bytes + class_bytes)
    if estimated_bytes > CanonicalLimits().max_output_bytes:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="geometry.triangle_area_result_bytes",
            message="triangle area profile exceeds the canonical output-byte limit",
        )


def compute_triangle_area_profile(
    configuration: PointConfiguration,
) -> TriangleAreaProfileResult:
    """Return the complete triangle-area profile of a planar configuration.

    For every triple of points, compute the exact unsigned triangle area
    using the cross-product formula. Points must be 2-dimensional.
    """
    if len(configuration.points[0].coordinates) != 2:
        raise OperationDomainValidationError(
            location=("configuration",),
            code="geometry.triangle_area_planar_configuration",
            message="triangle area profiles require exactly two coordinates per point",
        )
    _require_distinct_coordinates(configuration)
    _admit_triangle_area_result(configuration)
    points = configuration.points
    n = len(points)

    entries: list[TriangleAreaEntry] = []
    area_to_triples: dict[Fraction, list[tuple[int, int, int]]] = {}
    admitted_areas: list[tuple[tuple[int, int, int], Fraction]] = []

    for i, j, k in combinations(range(n), 3):
        coords_i = [c.as_fraction() for c in points[i].coordinates]
        coords_j = [c.as_fraction() for c in points[j].coordinates]
        coords_k = [c.as_fraction() for c in points[k].coordinates]

        # Signed area = 0.5 * |cross product|
        # cross = (x_j - x_i) * (y_k - y_i) - (x_k - x_i) * (y_j - y_i)
        dx1 = coords_j[0] - coords_i[0]
        dy1 = coords_j[1] - coords_i[1]
        dx2 = coords_k[0] - coords_i[0]
        dy2 = coords_k[1] - coords_i[1]

        cross = dx1 * dy2 - dx2 * dy1
        area = abs(cross) / 2

        triple = (i, j, k)
        numerator = format_canonical_integer(area.numerator)
        denominator = format_canonical_integer(area.denominator)
        if (
            len(numerator.lstrip("-")) > MAX_CANONICAL_RATIONAL_DIGITS
            or len(denominator) > MAX_CANONICAL_RATIONAL_DIGITS
        ):
            raise OperationDomainValidationError(
                location=("configuration",),
                code="geometry.triangle_area_result_bound",
                message=(
                    "a derived triangle area exceeds the canonical rational "
                    f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
                ),
            )
        admitted_areas.append((triple, area))

    for triple, area in admitted_areas:
        entries.append(
            TriangleAreaEntry(
                indices=triple,
                area=CanonicalRational.from_fraction(area),
            )
        )
        area_to_triples.setdefault(area, []).append(triple)

    # Build sorted area classes
    area_classes = tuple(
        (
            CanonicalRational.from_fraction(a),
            tuple(area_to_triples[a]),
        )
        for a in sorted(area_to_triples.keys())
    )

    return TriangleAreaProfileResult(
        configuration=configuration,
        entries=tuple(entries),
        area_classes=area_classes,
    )
