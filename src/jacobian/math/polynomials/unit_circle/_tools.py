"""Public declarations for exact unit-circle polynomial operations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.polynomials.unit_circle._models import (
    UnitCircleArcEnergyRequest,
    UnitCircleArcEnergyResult,
)
from jacobian.math.polynomials.unit_circle.operations import unit_circle_arc_energy


def _run_arc_energy(request: UnitCircleArcEnergyRequest) -> UnitCircleArcEnergyResult:
    return unit_circle_arc_energy(request)


POLYNOMIAL_ONE_PLUS_Z = {
    "polynomial": {
        "domain": "QQ",
        "variables": ["z"],
        "polynomial": {
            "terms": [
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [1]},
                {"coefficient": {"num": "1", "den": "1"}, "exponents": [0]},
            ]
        },
    },
    "start_turn": {"num": "-1", "den": "4"},
    "end_turn": {"num": "1", "den": "4"},
}


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="polynomial.unit_circle.arc_energy.compute",
        title="Compute exact unit-circle arc energy",
        description=(
            "Compute the exact normalized energy integral of a bounded rational "
            "polynomial on an oriented arc with unwrapped rational turns."
        ),
        request_type=UnitCircleArcEnergyRequest,
        result_type=UnitCircleArcEnergyResult,
        run=_run_arc_energy,
        tags=("polynomial", "unit-circle", "integral", "exact"),
        examples=(
            OperationExample(
                name="one_plus_z_right_semicircle",
                description="Energy of 1+z on the right semicircle.",
                input=POLYNOMIAL_ONE_PLUS_Z,
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
