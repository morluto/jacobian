"""Exact measure certificates over replayed visibility kernels (#974).

The kernel result's relied-upon relations (simplicity, counterclockwise
order, half-plane derivation, feasible boundary, hull, areas, ratios) are
replayed against the bound polygon before any measure is certified; the
profile then extends the result with exact edge lengths, rational
perimeters where every edge is rational, and caller-specified exact
comparisons.  Irrational lengths are never promoted: the squared length
stays exact while the length and any dependent perimeter stay absent.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, isqrt

from jacobian._exact import CanonicalRational, canonical_rational_component_digits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry._models import RationalPoint2D, _is_simple_ring
from jacobian.math.geometry.operations import convex_hull_points
from jacobian.math.geometry.polygon_kernel._kernel import (
    _canonical_hull,
    _evaluate,
    _feasible_boundary_intersections,
    _fraction_half_plane,
    _fraction_point,
    oriented_half_planes,
    polygon_signed_area,
)
from jacobian.math.geometry.polygon_kernel._models import (
    MAX_HALF_PLANE_COEFFICIENT_DIGITS,
    MAX_INTERSECTION_COMPONENT_DIGITS,
    MAX_KERNEL_COORDINATE_DIGITS,
    MAX_KERNEL_FEASIBILITY_WORK,
    MAX_KERNEL_SOURCE_VERTICES,
    EdgeLengthEntry,
    KernelPolygon,
    MeasureCertificateResult,
    MeasureComparison,
    MeasureComparisonOutcome,
    PolygonKernelResult,
    RingMeasureProfile,
    _evaluate_comparison,
)

__all__ = ["measure_certificate", "verify_measure_certificate"]


def _reject_measure(message: str) -> None:
    raise OperationDomainValidationError(
        location=("kernel",),
        code="geometry.measure_certificate_not_admitted",
        message=message,
    )


def _admit_measure_kernel(kernel: PolygonKernelResult) -> None:
    """Mirror the kernel admission envelope before replaying its relations."""

    points = kernel.polygon.points
    if not 3 <= len(points) <= MAX_KERNEL_SOURCE_VERTICES:
        _reject_measure(
            "kernel source vertices exceed the "
            f"{MAX_KERNEL_SOURCE_VERTICES}-vertex visibility-kernel bound"
        )
    max_coordinate_digits = max(
        canonical_rational_component_digits(component)
        for point in points
        for component in (point.x, point.y)
    )
    if max_coordinate_digits > MAX_KERNEL_COORDINATE_DIGITS:
        _reject_measure(
            "polygon coordinates exceed the "
            f"{MAX_KERNEL_COORDINATE_DIGITS}-digit visibility-kernel bound"
        )
    coefficient_digits = max(
        canonical_rational_component_digits(value)
        for half_plane in kernel.half_planes
        for value in (half_plane.a, half_plane.b, half_plane.c)
    )
    if coefficient_digits > MAX_HALF_PLANE_COEFFICIENT_DIGITS:
        _reject_measure(
            "oriented half-plane coefficients exceed the "
            f"{MAX_HALF_PLANE_COEFFICIENT_DIGITS}-digit bound"
        )
    if 8 * coefficient_digits + 8 > MAX_INTERSECTION_COMPONENT_DIGITS:
        _reject_measure(
            "a boundary-line intersection can exceed the "
            f"{MAX_INTERSECTION_COMPONENT_DIGITS}-digit component bound"
        )
    vertex_count = len(points)
    if (
        comb(vertex_count, 2) * vertex_count * coefficient_digits * coefficient_digits
        > MAX_KERNEL_FEASIBILITY_WORK
    ):
        _reject_measure(
            f"visibility-kernel feasibility work exceeds {MAX_KERNEL_FEASIBILITY_WORK}"
        )


def _domain_mismatch(message: str) -> OperationDomainValidationError:
    return OperationDomainValidationError(
        location=("kernel",),
        code="geometry.measure_certificate_kernel_mismatch",
        message=message,
    )


def _replay_kernel_relations(
    kernel: PolygonKernelResult,
) -> tuple[tuple[RationalPoint2D, ...], tuple[RationalPoint2D, ...]]:
    """Replay every relied-upon kernel relation; return boundary and hull."""

    points = kernel.polygon.points
    if not _is_simple_ring(points):
        raise _domain_mismatch("the bound polygon must be a simple polygon")
    if polygon_signed_area(points) <= 0:
        raise _domain_mismatch(
            "the bound polygon must use counterclockwise cyclic order"
        )
    half_planes = oriented_half_planes(KernelPolygon(points=points))
    if half_planes != kernel.half_planes:
        raise _domain_mismatch("the half-planes must match the bound polygon edges")
    coefficients = tuple(_fraction_half_plane(item) for item in half_planes)
    feasible = _feasible_boundary_intersections(half_planes)
    boundary = _canonical_hull(feasible)
    claimed_boundary = tuple(row.point for row in kernel.kernel_boundary)
    if boundary != claimed_boundary:
        raise _domain_mismatch(
            "the kernel boundary must match the feasible intersection hull"
        )
    for row, point in zip(kernel.kernel_boundary, boundary, strict=True):
        active = tuple(
            edge_index
            for edge_index, half_plane in enumerate(coefficients)
            if _evaluate(half_plane, _fraction_point(point)) == 0
        )
        if row.point != point or row.active_edge_indices != active:
            raise _domain_mismatch(
                "every boundary point must carry its tight edge indices"
            )
    hull = convex_hull_points(points).points
    if hull != kernel.convex_hull.points:
        raise _domain_mismatch("the hull must match the bound polygon")
    polygon_area = polygon_signed_area(points)
    kernel_area = polygon_signed_area(boundary)
    hull_area = polygon_signed_area(hull)
    for claimed, replayed, name in (
        (kernel.polygon_area, polygon_area, "polygon"),
        (kernel.kernel_area, kernel_area, "kernel"),
        (kernel.convex_hull_area, hull_area, "hull"),
    ):
        if claimed.as_fraction() != replayed:
            raise _domain_mismatch(f"the {name} area must match its boundary")
    for claimed, replayed, name in (
        (
            kernel.kernel_to_polygon_area_ratio,
            kernel_area / polygon_area,
            "kernel-to-polygon",
        ),
        (
            kernel.polygon_to_hull_area_ratio,
            polygon_area / hull_area,
            "polygon-to-hull",
        ),
    ):
        if claimed.as_fraction() != replayed:
            raise _domain_mismatch(f"the {name} area ratio must match its areas")
    return boundary, hull


def _squared_length(first: RationalPoint2D, second: RationalPoint2D) -> Fraction:
    delta_x = first.x.as_fraction() - second.x.as_fraction()
    delta_y = first.y.as_fraction() - second.y.as_fraction()
    return delta_x * delta_x + delta_y * delta_y


def _exact_length(squared: Fraction) -> CanonicalRational | None:
    """Return the exact rational root of a squared length, if rational."""

    if squared == 0:
        return CanonicalRational.from_fraction(Fraction(0))
    root_num, root_den = isqrt(squared.numerator), isqrt(squared.denominator)
    if root_num * root_num != squared.numerator:
        return None
    if root_den * root_den != squared.denominator:
        return None
    return CanonicalRational.from_fraction(Fraction(root_num, root_den))


def _ring_measures(vertices: tuple[RationalPoint2D, ...]) -> RingMeasureProfile:
    """Measure one cyclic boundary with at least three vertices."""

    if len(vertices) < 3:
        raise RuntimeError("a measured ring has at least three vertices")
    squared_lengths = [
        _squared_length(vertices[index], vertices[(index + 1) % len(vertices)])
        for index in range(len(vertices))
    ]
    edges = tuple(
        EdgeLengthEntry(
            squared_length=CanonicalRational.from_fraction(squared),
            exact_length=_exact_length(squared),
        )
        for squared in squared_lengths
    )
    total = Fraction(0)
    for entry in edges:
        if entry.exact_length is None:
            return RingMeasureProfile(vertices=vertices, edges=edges, perimeter=None)
        total += entry.exact_length.as_fraction()
    return RingMeasureProfile(
        vertices=vertices,
        edges=edges,
        perimeter=CanonicalRational.from_fraction(total),
    )


def measure_certificate(
    kernel: PolygonKernelResult,
    comparisons: tuple[MeasureComparison, ...] = (),
) -> MeasureCertificateResult:
    """Certify exact lengths, perimeters, and comparisons over a kernel result."""

    _admit_measure_kernel(kernel)
    boundary, hull = _replay_kernel_relations(kernel)
    polygon_measures = _ring_measures(kernel.polygon.points)
    hull_measures = _ring_measures(hull)
    if kernel.kernel_dimension == "POLYGON":
        kernel_measures = _ring_measures(boundary)
        segment_length = None
    elif kernel.kernel_dimension == "SEGMENT":
        (first, second) = boundary
        squared = _squared_length(first, second)
        kernel_measures = RingMeasureProfile(
            vertices=boundary, edges=(), perimeter=None
        )
        segment_length = _exact_length(squared)
    elif kernel.kernel_dimension == "POINT":
        kernel_measures = RingMeasureProfile(
            vertices=boundary,
            edges=(),
            perimeter=CanonicalRational.from_fraction(Fraction(0)),
        )
        segment_length = None
    else:
        kernel_measures = RingMeasureProfile(vertices=(), edges=(), perimeter=None)
        segment_length = None
    scalars: dict[str, CanonicalRational | None] = {
        "POLYGON_AREA": kernel.polygon_area,
        "KERNEL_AREA": kernel.kernel_area,
        "HULL_AREA": kernel.convex_hull_area,
        "POLYGON_PERIMETER": polygon_measures.perimeter,
        "HULL_PERIMETER": hull_measures.perimeter,
        "KERNEL_PERIMETER": kernel_measures.perimeter,
        "KERNEL_TO_POLYGON_AREA_RATIO": kernel.kernel_to_polygon_area_ratio,
        "POLYGON_TO_HULL_AREA_RATIO": kernel.polygon_to_hull_area_ratio,
    }
    outcomes: list[MeasureComparisonOutcome] = []
    for comparison in comparisons:
        left = scalars[comparison.left]
        right = scalars[comparison.right]
        if left is None or right is None:
            raise OperationDomainValidationError(
                location=("comparisons",),
                code="geometry.measure_comparison_unavailable",
                message="every compared measure must be present (rational)",
            )
        difference, holds = _evaluate_comparison(
            left.as_fraction(),
            right.as_fraction(),
            comparison.operator,
            comparison.expected_difference.as_fraction()
            if comparison.expected_difference is not None
            else None,
        )
        outcomes.append(
            MeasureComparisonOutcome(
                comparison=comparison,
                difference=CanonicalRational.from_fraction(difference),
                holds=holds,
            )
        )
    return MeasureCertificateResult._from_kernel(
        kernel=kernel,
        polygon_measures=polygon_measures,
        hull_measures=hull_measures,
        kernel_measures=kernel_measures,
        kernel_segment_length=segment_length,
        comparisons=tuple(outcomes),
    )


def verify_measure_certificate(claim: MeasureCertificateResult) -> bool:
    """Check a claimed measure certificate by recomputing it from its kernel."""
    try:
        return (
            measure_certificate(
                claim.kernel,
                tuple(outcome.comparison for outcome in claim.comparisons),
            )
            == claim
        )
    except OperationDomainValidationError:
        return False


__all__ = ["measure_certificate", "verify_measure_certificate"]
