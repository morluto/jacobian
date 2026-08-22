"""Lattice-polytope operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.lattice_polytopes._models import (
    CountLatticePointsResult,
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
        "Given a bounded rational polytope in V-representation (vertices) "
        "or H-representation (half-spaces) for ambient dimension d <= 4, "
        "enumerate every lattice (integer) point inside it. The facets of "
        "the convex hull are built with exact rational linear algebra and "
        "every integer point in the bounding box is tested against the "
        "exact half-space inequalities. A V-representation must be "
        "full-dimensional: its vertices must affinely span the ambient "
        "dimension, so lower-dimensional hulls (for example a segment in "
        "3-D space) are rejected.",
        LatticePolytopeRequest,
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
        "Given a bounded rational polytope in V-representation (vertices) "
        "or H-representation (half-spaces) for ambient dimension d <= 4, "
        "count the lattice (integer) points inside it without listing them. "
        "The count is exact: the facet half-spaces are built with exact "
        "rational linear algebra and every integer point in the bounding "
        "box is tested against the exact inequalities. A V-representation "
        "must be full-dimensional: its vertices must affinely span the "
        "ambient dimension, so lower-dimensional hulls (for example a "
        "segment in 3-D space) are rejected.",
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
