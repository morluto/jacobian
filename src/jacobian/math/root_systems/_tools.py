"""Root system operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.root_systems._models import (
    CartanMatrixRequest,
    RootSystemDataResult,
)
from jacobian.math.root_systems._operations import compute_root_system_data


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
        version="1",
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


# A2 Cartan matrix: [[2, -1], [-1, 2]]
_A2 = {"matrix": [[2, -1], [-1, 2]]}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "root_system.positive_roots.compute",
        "Compute positive roots from a Cartan matrix",
        "Compute all positive roots of a finite crystallographic root "
        "system from its Cartan matrix, using closure under simple "
        "reflections. Returns simple and positive roots plus highest-root "
        "and Coxeter data for each irreducible component.",
        CartanMatrixRequest,
        RootSystemDataResult,
        compute_root_system_data,
        "algebra",
        "root-system",
        "exact",
        examples=(
            example(
                "a2_cartan",
                "Compute root system data for A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                {"matrix": _A2["matrix"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
