"""Root system operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.root_systems._models import (
    MAX_RANK,
    CartanMatrixRequest,
    RootSystemDataResult,
    SimpleReflectionRequest,
    SimpleReflectionResult,
    WeylGroupOrderResult,
)
from jacobian.math.groups.root_systems.operations import (
    root_system_data,
    simple_reflection,
    weyl_group_order,
)


def _run_root_system_data(request: CartanMatrixRequest) -> RootSystemDataResult:
    return root_system_data(request.matrix)


def _run_simple_reflection(request: SimpleReflectionRequest) -> SimpleReflectionResult:
    return simple_reflection(request.matrix, request.vector, request.simple_index)


def _run_weyl_group_order(request: CartanMatrixRequest) -> WeylGroupOrderResult:
    return weyl_group_order(request.matrix)


_A2 = {
    "matrix": {
        "matrix": {
            "domain": "ZZ",
            "row_count": 2,
            "column_count": 2,
            "entries": [["2", "-1"], ["-1", "2"]],
        },
        "simple_root_axis": [0, 1],
    }
}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="root_system.positive_roots.compute",
        title="Compute positive roots from a Cartan matrix",
        description="Compute all positive roots of a finite crystallographic root "
        "system from its Cartan matrix, using closure under simple "
        "reflections. Returns simple and positive roots plus highest-root "
        "and Coxeter data for each irreducible component.",
        request_type=CartanMatrixRequest,
        result_type=RootSystemDataResult,
        run=_run_root_system_data,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_cartan",
                description="Compute root system data for A2; "
                "the matrix must be a valid finite-type Cartan matrix.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.simple_reflection.compute",
        title="Apply a simple reflection to a root lattice vector",
        description="Apply the simple reflection s_i to a vector in the root lattice "
        "of a finite crystallographic root system defined by its Cartan "
        f"matrix of rank at most {MAX_RANK}. The vector uses that same "
        "simple-root axis and the index is zero-based.",
        request_type=SimpleReflectionRequest,
        result_type=SimpleReflectionResult,
        run=_run_simple_reflection,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_reflection",
                description="Apply s_0 to the simple root alpha_0 in A2. The matrix is "
                "a finite-type generalized Cartan matrix, the vector has its "
                "two simple-root coordinates, and index 0 is below its rank.",
                input={"matrix": _A2["matrix"], "vector": [1, 0], "simple_index": 0},
            ),
        ),
    ),
    MathTool(
        operation_id="root_system.weyl_group_order.compute",
        title="Compute the exact order of a Weyl group",
        description="Compute the exact order of the Weyl group of a finite "
        "crystallographic root system from its Cartan matrix. The kernel "
        "constructs the bounded complete signed-root action and uses SymPy's "
        "Schreier-Sims order algorithm; it never enumerates Weyl-group elements.",
        request_type=CartanMatrixRequest,
        result_type=WeylGroupOrderResult,
        run=_run_weyl_group_order,
        tags=("algebra", "root-system", "exact"),
        examples=(
            OperationExample(
                name="a2_weyl_group_order",
                description="Compute the order of the A2 Weyl group, which is 6; "
                "the matrix must be a finite-type generalized Cartan "
                "matrix of rank at most 8.",
                input={"matrix": _A2["matrix"]},
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
