"""Root system operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.root_systems._models import (
    CartanMatrixRequest,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    WeylGroupOrderResult,
)
from jacobian.math.root_systems._operations import (
    compute_root_system_data,
    compute_simple_reflection,
    compute_weyl_group_order,
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
    _op(
        "root_system.simple_reflection.compute",
        "Apply a simple reflection to a root lattice vector",
        "Apply the simple reflection s_i to a vector in the root lattice "
        "of a finite crystallographic root system defined by its Cartan "
        "matrix.",
        SimpleReflectionRequest,
        SimpleReflectionResult,
        compute_simple_reflection,
        "algebra",
        "root-system",
        "exact",
        examples=(
            example(
                "a2_reflection",
                "Apply s_0 to the simple root alpha_0 in A2.",
                {"matrix": _A2["matrix"], "vector": [1, 0], "simple_index": 0},
            ),
        ),
    ),
    _op(
        "root_system.weyl_group_order.compute",
        "Compute the exact order of a Weyl group",
        "Compute the exact order of the Weyl group of a finite "
        "crystallographic root system from its Cartan matrix. The kernel "
        "constructs the bounded complete signed-root action and uses SymPy's "
        "Schreier-Sims order algorithm; it never enumerates Weyl-group elements.",
        CartanMatrixRequest,
        WeylGroupOrderResult,
        compute_weyl_group_order,
        "algebra",
        "root-system",
        "exact",
        examples=(
            example(
                "a2_weyl_group_order",
                "Compute the order of the A2 Weyl group, which is 6; "
                "the matrix must be a finite-type generalized Cartan "
                "matrix of rank at most 8.",
                {"matrix": _A2["matrix"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
