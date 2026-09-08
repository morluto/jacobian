"""Triangle area profile kernel."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from pydantic_core import PydanticCustomError

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.canonical import format_canonical_integer
from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.geometry.exact._models import PointConfiguration
from jacobian.math.geometry.exact.triangle_area_profile._models import (
    TriangleAreaEntry,
    TriangleAreaProfileResult,
    _require_distinct_coordinates,
)

__all__ = ["compute_triangle_area_profile", "verify_triangle_area_profile"]


MAX_TRANSLATION_BIT_WORK = 200_000_000


def _admit_triangle_area_result(
    configuration: PointConfiguration,
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Reject configurations whose complete profile cannot fit the wire limit."""
    points = configuration.points
    triangle_count = len(points) * (len(points) - 1) * (len(points) - 2) // 6
    if triangle_count == 0:
        return ()
    coordinate_widths = sorted(
        (
            max(
                len(format_canonical_integer(abs(coord.num))),
                len(format_canonical_integer(coord.den)),
            )
            for point in points
            for coord in point.coordinates
        ),
        reverse=True,
    )
    # A cross-product term can retain denominator factors from all six
    # coordinates in a triple; subtraction and the factor 1/2 add carry
    # digits. Reserve that complete factor product before enumeration.
    derived_digits = sum(coordinate_widths[:6], 0) + 2
    if derived_digits <= MAX_CANONICAL_RATIONAL_DIGITS:
        return tuple(
            (point.coordinates[0].as_fraction(), point.coordinates[1].as_fraction())
            for point in points
        )

    # Try a common-origin translation before refusing large absolute heights.
    # Rational subtraction uses two numerator/denominator products and a gcd.
    # The quadratic bit-product charge covers those products and Euclidean
    # reduction; integral translations have linear rather than quadratic cost.
    origin = points[0].coordinates
    work = 0
    for point in points:
        for axis, coordinate in enumerate(point.coordinates):
            anchor = origin[axis]
            a = max(1, abs(coordinate.num).bit_length())
            b = coordinate.den.bit_length()
            c = max(1, abs(anchor.num).bit_length())
            d = anchor.den.bit_length()
            numerator_bits = max(a + d, c + b) + 1
            denominator_bits = b + d
            work += 16 * (a * d + c * b + b * d + numerator_bits * denominator_bits)
    if work > MAX_TRANSLATION_BIT_WORK:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="geometry.triangle_area_result_bound",
            message=(
                "a derived triangle area exceeds the canonical rational "
                f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit bound"
            ),
        )
    anchor_x, anchor_y = origin[0].as_fraction(), origin[1].as_fraction()
    translated = tuple(
        (
            point.coordinates[0].as_fraction() - anchor_x,
            point.coordinates[1].as_fraction() - anchor_y,
        )
        for point in points
    )
    # A bit-length decimal upper bound avoids formatting an oversized private
    # difference. Translation is reused by every triple, preserving all areas.
    widths = sorted(
        (
            max(abs(value.numerator).bit_length(), value.denominator.bit_length())
            * 30_103
            // 100_000
            + 1
            for row in translated
            for value in row
        ),
        reverse=True,
    )
    if sum(widths[:6]) + 2 > MAX_CANONICAL_RATIONAL_DIGITS:
        raise OperationResourceAdmissionError(
            location=("configuration",),
            code="geometry.triangle_area_result_bound",
            message=f"a derived triangle area exceeds the canonical rational {MAX_CANONICAL_RATIONAL_DIGITS}-digit bound",
        )
    return translated


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
    try:
        _require_distinct_coordinates(configuration)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("configuration",), code=exc.type, message=exc.message()
        ) from exc
    coordinates = _admit_triangle_area_result(configuration)
    points = configuration.points
    n = len(points)

    entries: list[TriangleAreaEntry] = []
    area_to_triples: dict[Fraction, list[tuple[int, int, int]]] = {}
    admitted_areas: list[tuple[tuple[int, int, int], Fraction]] = []

    for i, j, k in combinations(range(n), 3):
        coords_i, coords_j, coords_k = coordinates[i], coordinates[j], coordinates[k]

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
            raise OperationResourceAdmissionError(
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


def verify_triangle_area_profile(claim: TriangleAreaProfileResult) -> bool:
    """Verify a serialized triangle-area profile against its source."""
    if not isinstance(claim, TriangleAreaProfileResult):
        return False
    try:
        return compute_triangle_area_profile(claim.configuration) == claim
    except OperationResourceAdmissionError:
        raise
    except OperationDomainValidationError:
        return False
