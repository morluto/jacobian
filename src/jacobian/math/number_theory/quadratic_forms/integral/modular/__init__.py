"""Quadratic polynomial values and operations over finite residue rings."""

from jacobian.math.number_theory.quadratic_forms.integral.modular._models import (
    ModularCoordinateVector,
    ModularEvaluationRequest,
    ModularInteger,
    ModularQuadraticCrossTerm,
    ModularQuadraticPolynomial,
    ModularQuadraticReduction,
    ModularReductionRequest,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber import (
    ModularQuadraticFiber,
    ModularQuadraticFiberRequest,
    compute_modular_quadratic_fiber,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.operations import (
    evaluate_modular_form,
    reduce_integral_form_modulus,
)

__all__ = [
    "ModularCoordinateVector",
    "ModularEvaluationRequest",
    "ModularInteger",
    "ModularQuadraticCrossTerm",
    "ModularQuadraticFiber",
    "ModularQuadraticFiberRequest",
    "ModularQuadraticPolynomial",
    "ModularQuadraticReduction",
    "ModularReductionRequest",
    "compute_modular_quadratic_fiber",
    "evaluate_modular_form",
    "reduce_integral_form_modulus",
]
