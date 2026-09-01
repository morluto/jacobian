"""Public declaration for exact rational polygon visibility kernels."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polygon_kernel._models import (
    PolygonKernelRequest,
    PolygonKernelResult,
)
from jacobian.math.geometry.polygon_kernel.operations import (
    visibility_kernel,
)


def _run_visibility_kernel(request: PolygonKernelRequest) -> PolygonKernelResult:
    return visibility_kernel(request.polygon)


_NAKANO_PENTAGON = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "4620", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "-4620", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "-385", "den": "1"}},
    {"x": {"num": "22176", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "385", "den": "1"}},
]

TOOLS = (
    MathTool(
        operation_id="geometry.polygon.visibility_kernel.compute",
        title="Reconstruct an exact polygon visibility kernel",
        description=(
            "Intersect the closed left half-plane of each edge of one bounded "
            "simple CCW rational polygon. Return source-bound oriented inequalities "
            "and vertex turns; the canonical empty, point, segment, or polygon "
            "kernel; the polygon convex hull; exact polygon, kernel, and hull "
            "areas; and rational area ratios. Admission allows 64 vertices and 64 "
            "digits per coordinate component, then bounds pairwise feasibility "
            "work, coefficient/intersection growth, and output before expansion. "
            "No perimeter or theorem-level claim."
        ),
        request_type=PolygonKernelRequest,
        result_type=PolygonKernelResult,
        run=_run_visibility_kernel,
        tags=("geometry", "polygon", "visibility-kernel", "exact"),
        examples=(
            OperationExample(
                name="published_pentagon_kernel",
                description=(
                    "Reconstruct the exact five-vertex kernel and rational area "
                    "profile of Nakano's counterclockwise pentagon."
                ),
                input={"polygon": {"points": _NAKANO_PENTAGON}},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
