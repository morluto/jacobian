"""Public declaration for exact rational polygon visibility kernels."""

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polygon_kernel._models import (
    MeasureCertificateRequest,
    MeasureCertificateResult,
    PolygonKernelRequest,
    PolygonKernelResult,
)
from jacobian.math.geometry.polygon_kernel.operations import (
    measure_certificate,
    visibility_kernel,
)


def _run_visibility_kernel(request: PolygonKernelRequest) -> PolygonKernelResult:
    return visibility_kernel(request.polygon)


def _run_measure_certificate(
    request: MeasureCertificateRequest,
) -> MeasureCertificateResult:
    return measure_certificate(request.kernel, request.comparisons)


_NAKANO_PENTAGON = [
    {"x": {"num": "0", "den": "1"}, "y": {"num": "4620", "den": "1"}},
    {"x": {"num": "0", "den": "1"}, "y": {"num": "-4620", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "-385", "den": "1"}},
    {"x": {"num": "22176", "den": "1"}, "y": {"num": "0", "den": "1"}},
    {"x": {"num": "23100", "den": "1"}, "y": {"num": "385", "den": "1"}},
]


def _rational(num: int, den: int = 1) -> dict[str, object]:
    return {"num": str(num), "den": str(den)}


def _kernel_point(x: int, y: int) -> dict[str, object]:
    return {"x": _rational(x), "y": _rational(y)}


_NAKANO_KERNEL: dict[str, object] = {
    "polygon": {
        "points": [
            _kernel_point(0, 4620),
            _kernel_point(0, -4620),
            _kernel_point(23100, -385),
            _kernel_point(22176, 0),
            _kernel_point(23100, 385),
        ]
    },
    "interior_half_plane_convention": "a*x+b*y+c>=0",
    "half_planes": [
        {"edge_index": 0, "a": _rational(9240), "b": _rational(0), "c": _rational(0)},
        {
            "edge_index": 1,
            "a": _rational(-4235),
            "b": _rational(23100),
            "c": _rational(106722000),
        },
        {
            "edge_index": 2,
            "a": _rational(-385),
            "b": _rational(-924),
            "c": _rational(8537760),
        },
        {
            "edge_index": 3,
            "a": _rational(-385),
            "b": _rational(924),
            "c": _rational(8537760),
        },
        {
            "edge_index": 4,
            "a": _rational(-4235),
            "b": _rational(-23100),
            "c": _rational(106722000),
        },
    ],
    "vertex_turns": [
        {"vertex_index": 0, "cross": _rational(213444000), "kind": "CONVEX"},
        {"vertex_index": 1, "cross": _rational(213444000), "kind": "CONVEX"},
        {"vertex_index": 2, "cross": _rational(12806640), "kind": "CONVEX"},
        {"vertex_index": 3, "cross": _rational(-711480), "kind": "REFLEX"},
        {"vertex_index": 4, "cross": _rational(12806640), "kind": "CONVEX"},
    ],
    "reflex_vertex_indices": [3],
    "kernel_dimension": "POLYGON",
    "kernel_boundary": [
        {"point": _kernel_point(0, -4620), "active_edge_indices": [0, 1]},
        {"point": _kernel_point(19800, -990), "active_edge_indices": [1, 3]},
        {"point": _kernel_point(22176, 0), "active_edge_indices": [2, 3]},
        {"point": _kernel_point(19800, 990), "active_edge_indices": [2, 4]},
        {"point": _kernel_point(0, 4620), "active_edge_indices": [0, 4]},
    ],
    "convex_hull": {
        "points": [
            _kernel_point(0, -4620),
            _kernel_point(23100, -385),
            _kernel_point(23100, 385),
            _kernel_point(0, 4620),
        ]
    },
    "polygon_area": _rational(115259760),
    "kernel_area": _rational(113430240),
    "convex_hull_area": _rational(115615500),
    "kernel_to_polygon_area_ratio": _rational(62, 63),
    "polygon_to_hull_area_ratio": _rational(324, 325),
}

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
    MathTool(
        operation_id="geometry.polygon.measure_certificate.compute",
        title="Certify exact polygon lengths, perimeters, and comparisons",
        description=(
            "Replay every relied-upon half-plane, boundary point, hull "
            "vertex, and area of a polygon kernel result, then extend the "
            "profile with exact edge lengths (squared lengths always; roots "
            "only when rational), rational perimeters where every edge is "
            "rational, and caller-specified exact comparisons between scalar "
            "measures. Irrational lengths are never promoted: the squared "
            "length stays exact while the length and any dependent perimeter "
            "stay absent. Admission mirrors the kernel envelope."
        ),
        request_type=MeasureCertificateRequest,
        result_type=MeasureCertificateResult,
        run=_run_measure_certificate,
        tags=("geometry", "polygon", "measure-certificate", "exact"),
        discovery_terms=(
            "polygon perimeter",
            "kernel area ratio",
            "exact edge length",
            "visibility kernel certificate",
        ),
        examples=(
            OperationExample(
                name="published_pentagon_measures",
                description=(
                    "Certify Nakano's pentagon: integer perimeters 58212 and "
                    "56980, area ratio 62/63, and the strict kernel/polygon "
                    "area containment."
                ),
                input={
                    "kernel": _NAKANO_KERNEL,
                    "comparisons": [
                        {
                            "left": "KERNEL_AREA",
                            "right": "POLYGON_AREA",
                            "operator": "LT",
                        },
                        {
                            "left": "KERNEL_TO_POLYGON_AREA_RATIO",
                            "right": "POLYGON_TO_HULL_AREA_RATIO",
                            "operator": "LT",
                            "expected_difference": {
                                "num": "-262",
                                "den": "20475",
                            },
                        },
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
