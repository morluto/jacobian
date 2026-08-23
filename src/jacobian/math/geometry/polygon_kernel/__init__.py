"""Exact visibility kernels of simple rational polygons."""

from jacobian.math.geometry.polygon_kernel._models import (
    KernelBoundaryIntersection,
    KernelPolygon,
    OrientedEdgeHalfPlane,
    PolygonKernelRequest,
    PolygonKernelResult,
    PolygonVertexTurn,
)
from jacobian.math.geometry.polygon_kernel._operations import (
    compute_visibility_kernel,
)

__all__ = [
    "KernelBoundaryIntersection",
    "KernelPolygon",
    "OrientedEdgeHalfPlane",
    "PolygonKernelRequest",
    "PolygonKernelResult",
    "PolygonVertexTurn",
    "compute_visibility_kernel",
]
