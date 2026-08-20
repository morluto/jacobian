"""Incidence structure operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.incidence_structures._models import (
    DegreeProfileResult,
    IncidenceMatrixRequest,
    IncidenceMatrixResult,
)
from jacobian.math.incidence_structures._operations import (
    compute_degree_profile,
    compute_incidence_matrix,
)


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
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


_STRUCTURE = {
    "points": ["p1", "p2", "p3"],
    "block_ids": ["b1", "b2"],
    "blocks": [["p1", "p2"], ["p2", "p3"]],
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "incidence.matrix.compute",
        "Compute the incidence matrix",
        "Compute the exact 0/1 incidence matrix of a finite incidence "
        "structure, with labelled point rows and block columns.",
        IncidenceMatrixRequest,
        IncidenceMatrixResult,
        compute_incidence_matrix,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_structure",
                "Compute the incidence matrix of a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
    _op(
        "incidence.degree_profile.compute",
        "Compute point and block degree profiles",
        "Compute per-point degrees (number of blocks containing each point) "
        "and per-block degrees (number of points in each block), with total "
        "incidence count.",
        IncidenceMatrixRequest,
        DegreeProfileResult,
        compute_degree_profile,
        "combinatorics",
        "incidence",
        "exact",
        examples=(
            example(
                "triangle_degrees",
                "Compute degree profiles for a 3-point, 2-block structure; "
                "every block member must be a declared point.",
                {"incidence": _STRUCTURE},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
