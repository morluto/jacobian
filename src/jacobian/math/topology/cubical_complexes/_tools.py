"""Cubical complex operation declarations."""

from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
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
    try:
        return f_vector(request.cells)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message=str(exc),
        ) from exc


def _face_closure(request: FaceClosureRequest) -> FaceClosureResult:
    try:
        return face_closure(request.cells)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("cells",),
            code="cubical_complex.invalid_ambient_axis",
            message=str(exc),
        ) from exc


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
    MathTool(
        operation_id="cubical.f_vector.compute",
        title="Compute the f-vector of a cubical complex",
        description="Compute the f-vector (cell counts by dimension) and Euler "
        "characteristic of a finite cubical complex composed of "
        "elementary unit lattice cubes.",
        request_type=CubicalComplexRequest,
        result_type=FVectorResult,
        run=_f_vector,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="four_squares",
                description="Compute the f-vector of four unit squares forming a 2x2 grid; "
                "each interval must be unit length (b = a + 1).",
                input=_CELLS,
            ),
        ),
    ),
    MathTool(
        operation_id="cubical.face_closure.compute",
        title="Compute the face closure of a cubical complex",
        description="Compute the full face closure (all proper faces) of a set "
        "of elementary cubes, returning total cell count and "
        "cells by dimension.",
        request_type=FaceClosureRequest,
        result_type=FaceClosureResult,
        run=_face_closure,
        tags=("topology", "cubical", "exact"),
        examples=(
            OperationExample(
                name="single_square_closure",
                description="Compute the face closure of a single unit square; "
                "each interval must be unit length (b = a + 1).",
                input={"cells": [{"intervals": [[0, 1], [0, 1]]}]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
