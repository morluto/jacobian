"""Native exact visibility-kernel operation."""

from __future__ import annotations

from math import comb

from jacobian._exact import canonical_rational_component_digits
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.geometry.polygon_kernel._kernel import (
    compute_kernel_data,
    oriented_half_planes,
)
from jacobian.math.geometry.polygon_kernel._models import (
    MAX_HALF_PLANE_COEFFICIENT_DIGITS,
    MAX_INTERSECTION_COMPONENT_DIGITS,
    MAX_KERNEL_COORDINATE_DIGITS,
    MAX_KERNEL_FEASIBILITY_WORK,
    MAX_KERNEL_RESULT_CHARS,
    KernelPolygon,
    OrientedEdgeHalfPlane,
    PolygonKernelResult,
    _estimate_visibility_kernel_result_characters,
)


def _admit_visibility_kernel(
    polygon: KernelPolygon,
) -> tuple[OrientedEdgeHalfPlane, ...]:
    """Check derived work/output bounds and return prepared half-planes."""
    max_coordinate_digits = max(
        canonical_rational_component_digits(component)
        for point in polygon.points
        for component in (point.x, point.y)
    )
    if max_coordinate_digits > MAX_KERNEL_COORDINATE_DIGITS:
        raise OperationDomainValidationError(
            location=("polygon",),
            code="geometry.polygon_kernel.coordinate_digits",
            message=(
                "polygon coordinates exceed the "
                f"{MAX_KERNEL_COORDINATE_DIGITS}-digit visibility-kernel bound"
            ),
        )

    half_planes = oriented_half_planes(polygon)
    coefficient_digits = max(
        canonical_rational_component_digits(value)
        for half_plane in half_planes
        for value in (half_plane.a, half_plane.b, half_plane.c)
    )
    if coefficient_digits > MAX_HALF_PLANE_COEFFICIENT_DIGITS:
        raise OperationDomainValidationError(
            location=("polygon",),
            code="geometry.polygon_kernel.half_plane_digits",
            message=(
                "oriented half-plane coefficients exceed the "
                f"{MAX_HALF_PLANE_COEFFICIENT_DIGITS}-digit bound"
            ),
        )
    intersection_digits = 8 * coefficient_digits + 8
    if intersection_digits > MAX_INTERSECTION_COMPONENT_DIGITS:
        raise OperationDomainValidationError(
            location=("polygon",),
            code="geometry.polygon_kernel.intersection_digits",
            message=(
                "a boundary-line intersection can exceed the "
                f"{MAX_INTERSECTION_COMPONENT_DIGITS}-digit component bound"
            ),
        )

    vertex_count = len(polygon.points)
    estimated_result_chars = _estimate_visibility_kernel_result_characters(
        vertex_count,
        max_coordinate_digits,
        coefficient_digits,
        intersection_digits,
    )
    if estimated_result_chars > MAX_KERNEL_RESULT_CHARS:
        raise OperationDomainValidationError(
            location=("polygon",),
            code="geometry.polygon_kernel.result_size",
            message=(
                "visibility-kernel result can require "
                f"{estimated_result_chars} characters, exceeding the "
                f"{MAX_KERNEL_RESULT_CHARS}-character bound"
            ),
        )
    feasibility_work = (
        comb(vertex_count, 2) * vertex_count * coefficient_digits * coefficient_digits
    )
    if feasibility_work > MAX_KERNEL_FEASIBILITY_WORK:
        raise OperationDomainValidationError(
            location=("polygon",),
            code="geometry.polygon_kernel.feasibility_work",
            message=(
                "visibility-kernel feasibility work exceeds "
                f"{MAX_KERNEL_FEASIBILITY_WORK}"
            ),
        )
    return half_planes


def visibility_kernel(polygon: KernelPolygon) -> PolygonKernelResult:
    """Reconstruct a simple CCW polygon's closed visibility kernel exactly."""
    half_planes = _admit_visibility_kernel(polygon)
    data = compute_kernel_data(polygon, half_planes=half_planes)
    return PolygonKernelResult._from_kernel(polygon, data=data)


__all__ = ["visibility_kernel"]
