"""Quadratic polynomial values and operations over finite residue rings."""

from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularCoordinateVector,
    ModularEvaluationRequest,
    ModularInteger,
    ModularQuadraticPolynomial,
    ModularReductionRequest,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.operations import (
    evaluate_modular_form,
    reduce_integral_form_modulus,
)

__all__ = [
    "ModularCoordinateVector",
    "ModularEvaluationRequest",
    "ModularInteger",
    "ModularQuadraticPolynomial",
    "ModularReductionRequest",
    "evaluate_modular_form",
    "reduce_integral_form_modulus",
]
