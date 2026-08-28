"""Lattice-polytope operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polytope.lattice_points.enumerate",
        "Enumerate lattice points inside a bounded rational polytope",
        "Enumerate every lattice (integer) point of a bounded rational "
        "polytope in V- or H-representation for d <= 4, exactly. "
        "A V-representation must be full-dimensional (vertices must affinely "
        "span the ambient dimension); the supported exception is a "
        "one-dimensional input, accepted for every vertex family including "
        "a single point. A bounded empty H-system yields no points; every "
        "half-space needs a nonzero normal.",
        EnumerateLatticePointsRequest,
        EnumerateLatticePointsResult,
        enumerate_lattice_points,
        "polytope",
        "lattice",
        "exact",
        examples=(
            example(
                "unit_square_vertices",
                "Unit square [0,1]^2 has four lattice points. Requires "
                "vertices spanning the full ambient dimension: a "
                "lower-dimensional hull is rejected at validation.",
                {
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
    _op(
        "polytope.lattice_points.count",
        "Count lattice points inside a bounded rational polytope",
        "Count, exactly, the lattice (integer) points of a bounded "
        "rational polytope in V- or H-representation for d <= 4 without "
        "listing them. A V-representation must be full-dimensional "
        "(vertices must affinely span the ambient dimension); the supported "
        "exception is a one-dimensional input, accepted for every vertex "
        "family including a single point. A bounded empty H-system counts "
        "zero; every half-space needs a nonzero normal.",
        LatticePolytopeRequest,
        CountLatticePointsResult,
        count_lattice_points,
        "polytope",
        "lattice",
        "exact",
        examples=(
            example(
                "unit_square_halfspaces",
                "Unit square [0,1]^2 via half-spaces has four lattice "
                "points. Requires the H-representation to define a bounded "
                "polytope (normals positively spanning R^d); unbounded "
                "systems are rejected at validation.",
                {
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
