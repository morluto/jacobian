"""Native exact visibility-kernel operation and private claim verification."""

from jacobian.math.geometry.polygon_kernel._kernel import compute_kernel_data
from jacobian.math.geometry.polygon_kernel._models import (
    PolygonKernelRequest,
    PolygonKernelResult,
)


def compute_visibility_kernel(request: PolygonKernelRequest) -> PolygonKernelResult:
    """Reconstruct a simple CCW polygon's closed visibility kernel exactly."""

    data = compute_kernel_data(request.polygon)
    return PolygonKernelResult._from_kernel(request.polygon, data=data)


def _verify_polygon_kernel_result(result: PolygonKernelResult) -> bool:
    """Deliberately recompute one independently supplied kernel claim."""

    try:
        request = PolygonKernelRequest.model_validate(
            {"polygon": result.polygon.model_dump(mode="json")}
        )
    except (AttributeError, TypeError, ValueError):
        return False
    expected = compute_kernel_data(request.polygon)
    return (
        result.interior_half_plane_convention == expected.convention
        and result.half_planes == expected.half_planes
        and result.vertex_turns == expected.vertex_turns
        and result.reflex_vertex_indices == expected.reflex_vertex_indices
        and result.kernel_dimension == expected.dimension
        and result.kernel_boundary == expected.boundary
        and result.convex_hull == expected.convex_hull
        and result.polygon_area == expected.polygon_area
        and result.kernel_area == expected.kernel_area
        and result.convex_hull_area == expected.convex_hull_area
        and result.kernel_to_polygon_area_ratio == expected.kernel_to_polygon_area_ratio
        and result.polygon_to_hull_area_ratio == expected.polygon_to_hull_area_ratio
    )


__all__ = ["compute_visibility_kernel"]
