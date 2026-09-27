"""Catalog declaration for root-system weight/root quotient invariants."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.groups.root_systems._models import CartanMatrixRequest
from jacobian.math.groups.root_systems.connection_index._models import (
    RootSystemConnectionIndexResult,
)
from jacobian.math.groups.root_systems.connection_index.operations import (
    root_system_connection_index,
)


def _compute(
    request: CartanMatrixRequest,
) -> RootSystemConnectionIndexResult:
    return root_system_connection_index(request.matrix)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="root_system.connection_index.compute",
        title="Compute the weight-lattice quotient by the root lattice",
        description=(
            "Return the exact invariant factors and order of P/Q for finite "
            "crystallographic Cartan data. The Cartan matrix is the integral "
            "inclusion of the root lattice in fundamental-weight coordinates."
        ),
        request_type=CartanMatrixRequest,
        result_type=RootSystemConnectionIndexResult,
        run=_compute,
        tags=(
            "root-system",
            "root-lattice",
            "weight-lattice",
            "quotient",
            "smith-normal-form",
        ),
        discovery_terms=(
            "root system connection index",
            "weight lattice modulo root lattice",
            "fundamental group of a root system",
            "invariant factors of P/Q",
        ),
        examples=(
            OperationExample(
                name="type_a2_weight_root_quotient",
                description=(
                    "For type A2, the weight lattice modulo the root lattice "
                    "is cyclic of order three."
                ),
                input={
                    "matrix": {
                        "matrix": {
                            "domain": "ZZ",
                            "row_count": 2,
                            "column_count": 2,
                            "entries": [["2", "-1"], ["-1", "2"]],
                        },
                        "simple_root_axis": [0, 1],
                    }
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
