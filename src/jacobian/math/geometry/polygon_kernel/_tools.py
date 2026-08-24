"""Public declaration for exact rational polygon visibility kernels."""

from jacobian.catalog._examples import example
from jacobian.math.geometry._support import geometry_operation
from jacobian.math.geometry.polygon_kernel._models import (
    PolygonKernelRequest,
    PolygonKernelResult,
)
from jacobian.math.geometry.polygon_kernel._operations import (
    compute_visibility_kernel,
)

_NAKANO_PENTAGON = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "4620", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "-4620", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "-385", "den": "1"}},
    {"x": {"num": "22176", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "385", "den": "1"}},
]

TOOLS = (
    geometry_operation(
        "geometry.polygon.visibility_kernel.compute",
        "Reconstruct an exact polygon visibility kernel",
        (
            "Intersect the closed left half-plane of each edge of one bounded "
            "simple CCW rational polygon. Return source-bound oriented inequalities "
            "and vertex turns; the canonical empty, point, segment, or polygon "
            "kernel; the polygon convex hull; exact polygon, kernel, and hull "
            "areas; and rational area ratios. Admission allows 64 vertices and 64 "
            "digits per coordinate component, then bounds pairwise feasibility "
            "work, coefficient/intersection growth, and output before expansion. "
            "No perimeter or theorem-level claim."
        ),
        PolygonKernelRequest,
        PolygonKernelResult,
        compute_visibility_kernel,
        "geometry",
        "polygon",
        "visibility-kernel",
        "exact",
        examples=(
            example(
                "published_pentagon_kernel",
                (
                    "Reconstruct the exact five-vertex kernel and rational area "
                    "profile of Nakano's counterclockwise pentagon."
                ),
                {"polygon": {"points": _NAKANO_PENTAGON}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
