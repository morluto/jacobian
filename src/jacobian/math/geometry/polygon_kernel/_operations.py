"""Native exact visibility-kernel operation."""

from jacobian.math.geometry.polygon_kernel._kernel import compute_kernel_data
from jacobian.math.geometry.polygon_kernel._models import (
    PolygonKernelRequest,
    PolygonKernelResult,
)


def compute_visibility_kernel(request: PolygonKernelRequest) -> PolygonKernelResult:
    """Reconstruct a simple CCW polygon's closed visibility kernel exactly."""

    data = compute_kernel_data(request.polygon)
    # ``compute_kernel_data`` is the same deterministic replay used by the
    # result model. Avoid paying the O(n^3) replay twice in the trusted producer;
    # serialized/authored results still run the complete source-binding check.
    return PolygonKernelResult.model_construct(
        polygon=request.polygon,
        interior_half_plane_convention=data.convention,
        half_planes=data.half_planes,
        vertex_turns=data.vertex_turns,
        reflex_vertex_indices=data.reflex_vertex_indices,
        kernel_dimension=data.dimension,
        kernel_boundary=data.boundary,
        convex_hull=data.convex_hull,
        polygon_area=data.polygon_area,
        kernel_area=data.kernel_area,
        convex_hull_area=data.convex_hull_area,
        kernel_to_polygon_area_ratio=data.kernel_to_polygon_area_ratio,
        polygon_to_hull_area_ratio=data.polygon_to_hull_area_ratio,
    )


__all__ = ["compute_visibility_kernel"]
