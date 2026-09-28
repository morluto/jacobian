"""Complete fibers of finite-modulus quadratic polynomials."""

from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber._models import (
    ModularQuadraticFiber,
)
from jacobian.math.number_theory.quadratic_forms.integral.modular.fiber.operations import (
    compute_modular_quadratic_fiber,
)

__all__ = [
    "ModularQuadraticFiber",
    "compute_modular_quadratic_fiber",
]
