"""Lattice-polytope operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lattice_polytopes._models import (
    CountLatticePointsResult,
    EnumerateLatticePointsRequest,
    EnumerateLatticePointsResult,
    LatticePolytopeRequest,
)
from jacobian.math.lattice_polytopes._operations import (
    count_lattice_points,
    enumerate_lattice_points,
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
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


LATTICE_POLYTOPE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polytope.lattice_points.enumerate",
        "Enumerate lattice points inside a bounded rational polytope",
        "Enumerate every lattice (integer) point of a bounded rational "
        "polytope in V- or H-representation for dimension d <= 4, exactly. "
        "Facets come from exact rational linear algebra; every integer "
        "point in the bounding box is tested against the exact half-space "
        "inequalities. A V-representation must be full-dimensional: its "
        "vertices must affinely span the ambient dimension, so lower-"
        "dimensional hulls are rejected; the supported exception is a "
        "one-dimensional input, where every vertex family (including a "
        "single point) is processed exactly.",
        EnumerateLatticePointsRequest,
        EnumerateLatticePointsResult,
        enumerate_lattice_points,
        "polytope",
        "lattice",
        "exact",
        examples=(
            example(
                "unit_square_vertices",
                "Unit square [0,1]^2 has four lattice points.",
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
        "rational polytope in V- or H-representation for dimension "
        "d <= 4 without listing them. Facets come from exact rational "
        "linear algebra; every integer point in the bounding box is "
        "tested against the exact inequalities. A V-representation must "
        "be full-dimensional: its vertices must affinely span the "
        "ambient dimension, so lower-dimensional hulls are rejected; the "
        "supported exception is a one-dimensional input, where every "
        "vertex family (including a single point) is processed exactly.",
        LatticePolytopeRequest,
        CountLatticePointsResult,
        count_lattice_points,
        "polytope",
        "lattice",
        "exact",
        examples=(
            example(
                "unit_square_halfspaces",
                "Unit square [0,1]^2 via half-spaces has four lattice points.",
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

TOOLS = LATTICE_POLYTOPE_OPERATIONS

__all__ = ["TOOLS"]
