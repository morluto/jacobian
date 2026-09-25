"""Public constant cellular-sheaf construction operation."""

from __future__ import annotations

from jacobian.catalog.models import MathTool, MathTools, OperationExample
from jacobian.math.topology.cellular_sheaves._models import FiniteCellularSheaf
from jacobian.math.topology.cellular_sheaves.constants._models import (
    ConstantSheafRequest,
)
from jacobian.math.topology.cellular_sheaves.constants.operations import (
    constant_sheaf,
)


def _run(request: ConstantSheafRequest) -> FiniteCellularSheaf:
    return constant_sheaf(request)


_INTERVAL = {
    "vertices": ["a", "b"],
    "maximal_simplices": [["a", "b"]],
    "faces_by_dimension": [
        {"dimension": 0, "faces": [["a"], ["b"]]},
        {"dimension": 1, "faces": [["a", "b"]]},
    ],
    "dimension": 1,
    "f_vector": [2, 1],
    "closure_size": 3,
}


TOOLS: MathTools = (
    MathTool(
        operation_id="cellular_sheaf.constant.compute",
        title="Construct a constant cellular sheaf",
        description=(
            "Copy one based vector space over QQ or a bounded prime field to "
            "every nonempty simplex of a finite simplicial complex. Every "
            "face-to-coface restriction is the identity under the copied basis "
            "identifications. The result is a canonical cellular sheaf usable "
            "unchanged by section and cohomology operations."
        ),
        request_type=ConstantSheafRequest,
        result_type=FiniteCellularSheaf,
        run=_run,
        tags=("topology", "cellular-sheaf", "constant-sheaf", "exact"),
        discovery_terms=(
            "constant cellular sheaf",
            "locally constant vector-space stalks",
            "identity face restrictions",
        ),
        examples=(
            OperationExample(
                name="rank_one_constant_sheaf_on_interval",
                description=(
                    "Put the rational line on each vertex and edge of an "
                    "interval, with identity restrictions."
                ),
                input={
                    "complex": _INTERVAL,
                    "coefficient_field": "QQ",
                    "basis": ["e"],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
