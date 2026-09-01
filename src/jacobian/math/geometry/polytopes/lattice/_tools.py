"""Lattice-polytope operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.geometry.polytopes.lattice._models import (
    CountLatticePointsResult,
    EnumerateLatticePointsRequest,
    EnumerateLatticePointsResult,
    LatticePolytopeRequest,
)
from jacobian.math.geometry.polytopes.lattice.operations import (
    count_lattice_points as native_count_lattice_points,
)
from jacobian.math.geometry.polytopes.lattice.operations import (
    enumerate_lattice_points as native_enumerate_lattice_points,
)


def enumerate_lattice_points(
    request: EnumerateLatticePointsRequest,
) -> EnumerateLatticePointsResult:
    """Unpack a request and project the native lattice enumeration."""
    return native_enumerate_lattice_points(
        request.vertices,
        request.halfspaces,
        request.dimension_bound,
    )


def count_lattice_points(request: LatticePolytopeRequest) -> CountLatticePointsResult:
    """Unpack a request and project the native lattice count."""
    return native_count_lattice_points(
        request.vertices,
        request.halfspaces,
        request.dimension_bound,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polytope.lattice_points.enumerate",
        title="Enumerate lattice points inside a bounded rational polytope",
        description="Enumerate every lattice (integer) point of a bounded rational "
        "polytope in V- or H-representation for d <= 4, exactly. "
        "A V-representation must be full-dimensional (vertices must affinely "
        "span the ambient dimension); the supported exception is a "
        "one-dimensional input, accepted for every vertex family including "
        "a single point. A bounded empty H-system yields no points; every "
        "half-space needs a nonzero normal.",
        request_type=EnumerateLatticePointsRequest,
        result_type=EnumerateLatticePointsResult,
        run=enumerate_lattice_points,
        tags=("polytope", "lattice", "exact"),
        examples=(
            OperationExample(
                name="unit_square_vertices",
                description="Unit square [0,1]^2 has four lattice points. Requires "
                "vertices spanning the full ambient dimension: a "
                "lower-dimensional hull is rejected at validation.",
                input={
                    "vertices": [
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        },
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                        },
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="polytope.lattice_points.count",
        title="Count lattice points inside a bounded rational polytope",
        description="Count, exactly, the lattice (integer) points of a bounded "
        "rational polytope in V- or H-representation for d <= 4 without "
        "listing them. A V-representation must be full-dimensional "
        "(vertices must affinely span the ambient dimension); the supported "
        "exception is a one-dimensional input, accepted for every vertex "
        "family including a single point. A bounded empty H-system counts "
        "zero; every half-space needs a nonzero normal.",
        request_type=LatticePolytopeRequest,
        result_type=CountLatticePointsResult,
        run=count_lattice_points,
        tags=("polytope", "lattice", "exact"),
        examples=(
            OperationExample(
                name="unit_square_halfspaces",
                description="Unit square [0,1]^2 via half-spaces has four lattice "
                "points. Requires the H-representation to define a bounded "
                "polytope (normals positively spanning R^d); unbounded "
                "systems are rejected at validation.",
                input={
                    "halfspaces": [
                        {
                            "coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "offset": {"num": "1", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "offset": {"num": "0", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "offset": {"num": "1", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "-1", "den": "1"},
                            ],
                            "offset": {"num": "0", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
