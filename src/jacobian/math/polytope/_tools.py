"""Polytope operation ownership and declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polytope._models import (
    PolytopeVolumeRequest,
    PolytopeVolumeResult,
)
from jacobian.math.polytope._operations import compute_polytope_volume


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


POLYTOPE_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "polytope.volume.compute",
        "Compute the exact rational volume of a bounded polytope",
        "Compute the exact rational volume of a bounded rational polytope "
        "from its V-representation (vertices) or H-representation (half-spaces) "
        "for ambient dimension d <= 6, via triangulation and SymPy exact "
        "determinant-based simplex volume. Every half-space must carry a "
        "nonzero normal: rows whose coefficients are all zero are rejected.",
        PolytopeVolumeRequest,
        PolytopeVolumeResult,
        compute_polytope_volume,
        "polytope",
        "volume",
        "exact-rational",
        examples=(
            example(
                "unit_cube_vertices",
                "Unit cube [0,1]^2 split into two triangles (volume = 1).",
                {
                    "vertices": [
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "1", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                        {
                            "coordinates": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ]
                        },
                    ],
                },
            ),
            example(
                "unit_square_halfspaces",
                "Unit square [0,1]^2 as four half-spaces, each with a "
                "nonzero normal (volume = 1).",
                {
                    "halfspaces": [
                        {
                            "coefficients": [
                                {"num": "-1", "den": "1"},
                                {"num": "0", "den": "1"},
                            ],
                            "offset": {"num": "0", "den": "1"},
                        },
                        {
                            "coefficients": [
                                {"num": "1", "den": "1"},
                                {"num": "0", "den": "1"},
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
                        {
                            "coefficients": [
                                {"num": "0", "den": "1"},
                                {"num": "1", "den": "1"},
                            ],
                            "offset": {"num": "1", "den": "1"},
                        },
                    ],
                },
            ),
        ),
    ),
)


TOOLS = POLYTOPE_OPERATIONS

__all__ = ["TOOLS"]
