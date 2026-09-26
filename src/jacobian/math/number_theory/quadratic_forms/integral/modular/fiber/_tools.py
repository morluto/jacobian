"""Public complete-fiber operation for finite-modulus quadratic polynomials."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber._models import (
    ModularQuadraticFiber,
    ModularQuadraticFiberRequest,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber.operations import (
    compute_modular_quadratic_fiber,
)


def _run_fiber(request: ModularQuadraticFiberRequest) -> ModularQuadraticFiber:
    return compute_modular_quadratic_fiber(request.polynomial, request.target)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.modular_fiber.compute",
        title="Compute a complete finite-modulus quadratic fiber",
        description=(
            "Return every vector on the polynomial's exact ordered coordinate axis "
            "that evaluates to the target residue. Vectors are lexicographically "
            "ordered; an empty tuple is the complete empty fiber. The exhaustive "
            "search admits at most 100000 domain vectors, 20000000 scalar-work "
            "units, and 4000000 result digit-value units."
        ),
        request_type=ModularQuadraticFiberRequest,
        result_type=ModularQuadraticFiber,
        run=_run_fiber,
        tags=("quadratic-form", "modular", "fiber", "exact"),
        discovery_terms=(
            "solutions to quadratic polynomial modulo m",
            "complete fiber of modular quadratic form",
            "vectors with prescribed quadratic residue",
        ),
        examples=(
            OperationExample(
                name="mixed_polynomial_fiber_modulo_three",
                description="Find the complete zero fiber of x^2+xy+y^2 over Z/3Z.",
                input={
                    "polynomial": {
                        "modulus": "3",
                        "axis": ["x", "y"],
                        "diagonal_residues": ["1", "1"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "1"}],
                    },
                    "target": {"modulus": "3", "residue": "0"},
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
