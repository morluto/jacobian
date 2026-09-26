"""Public exact operations on finite-modulus quadratic polynomials."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.quadratic_forms.integral.modular import (
    operations as native,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularEvaluationRequest,
    ModularInteger,
    ModularQuadraticReduction,
    ModularReductionRequest,
)

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="quadratic_form.modulus_reduce.compute",
        title="Reduce an integral quadratic polynomial modulo an integer",
        description=(
            "Reduce each polynomial coefficient into the explicitly retained "
            "ring Z/mZ, preserving the ordered coordinate axis and polynomial "
            "monomial convention. The result is a polynomial presentation, "
            "not a canonical classification of functions on the finite module."
        ),
        request_type=ModularReductionRequest,
        result_type=ModularQuadraticReduction,
        run=native.reduce_integral_form_modulus,
        tags=("quadratic-form", "modular", "coefficient-map", "exact"),
        discovery_terms=(
            "reduce an integral quadratic form modulo m",
            "quadratic polynomial over Z/mZ",
            "finite modulus quadratic form coefficient reduction",
        ),
        examples=(
            OperationExample(
                name="composite_modulus_preserves_polynomial_coefficients",
                description="Reduce 2x^2+3xy-y^2 into Z/6Z without changing its axis.",
                input={
                    "form": {
                        "axis": ["x", "y"],
                        "diagonal_coefficients": ["2", "-1"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "3"}],
                    },
                    "modulus": "6",
                },
            ),
        ),
    ),
    MathTool(
        operation_id="quadratic_form.modular.evaluate.compute",
        title="Evaluate a finite-modulus quadratic polynomial",
        description=(
            "Evaluate a canonical residue polynomial on a coordinate vector "
            "with the same modulus and ordered axis, returning its residue in "
            "that parent ring."
        ),
        request_type=ModularEvaluationRequest,
        result_type=ModularInteger,
        run=native.evaluate_modular_form,
        tags=("quadratic-form", "modular", "evaluation", "exact"),
        discovery_terms=(
            "evaluate quadratic polynomial over Z/mZ",
            "quadratic form value modulo an integer",
            "evaluate finite ring quadratic polynomial at vector",
        ),
        examples=(
            OperationExample(
                name="evaluate_in_the_same_residue_ring",
                description="Evaluate the reduced polynomial at (4, 5) in Z/6Z.",
                input={
                    "polynomial": {
                        "modulus": "6",
                        "axis": ["x", "y"],
                        "diagonal_residues": ["2", "5"],
                        "cross_terms": [{"left": 0, "right": 1, "coefficient": "3"}],
                    },
                    "vector": {
                        "modulus": "6",
                        "axis": ["x", "y"],
                        "coordinates": ["4", "5"],
                    },
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
