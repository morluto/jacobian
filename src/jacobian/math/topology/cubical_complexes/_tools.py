"""Cubical complex operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.topology.cubical_complexes._models import (
    CubicalComplexRequest,
    FaceClosureRequest,
    FaceClosureResult,
    FVectorResult,
)
from jacobian.math.topology.cubical_complexes.operations import (
    f_vector,
    face_closure,
)


def _f_vector(request: CubicalComplexRequest) -> FVectorResult:
    return f_vector(request.cells)


def _face_closure(request: FaceClosureRequest) -> FaceClosureResult:
    return face_closure(request.cells)


def _op[RequestT: StrictModel, ResultT: StrictModel](
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


# A single 2D square: [(0,1),(0,1)] + [(0,1),(1,2)] + [(1,2),(0,1)] + [(1,2),(1,2)]
_CELLS = {
    "cells": [
        {"intervals": [[0, 1], [0, 1]]},
        {"intervals": [[0, 1], [1, 2]]},
        {"intervals": [[1, 2], [0, 1]]},
        {"intervals": [[1, 2], [1, 2]]},
    ]
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "cubical.f_vector.compute",
        "Compute the f-vector of a cubical complex",
        "Compute the f-vector (cell counts by dimension) and Euler "
        "characteristic of a finite cubical complex composed of "
        "elementary unit lattice cubes.",
        CubicalComplexRequest,
        FVectorResult,
        _f_vector,
        "topology",
        "cubical",
        "exact",
        examples=(
            example(
                "four_squares",
                "Compute the f-vector of four unit squares forming a 2x2 grid; "
                "each interval must be unit length (b = a + 1).",
                _CELLS,
            ),
        ),
    ),
    _op(
        "cubical.face_closure.compute",
        "Compute the face closure of a cubical complex",
        "Compute the full face closure (all proper faces) of a set "
        "of elementary cubes, returning total cell count and "
        "cells by dimension.",
        FaceClosureRequest,
        FaceClosureResult,
        _face_closure,
        "topology",
        "cubical",
        "exact",
        examples=(
            example(
                "single_square_closure",
                "Compute the face closure of a single unit square; "
                "each interval must be unit length (b = a + 1).",
                {"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
