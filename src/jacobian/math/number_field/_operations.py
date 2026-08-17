"""Domain-owned number field operations."""

from __future__ import annotations

from jacobian.math.number_field import discriminant
from jacobian.math.number_field._models import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    disc = discriminant(list(request.coefficients_descending), request.variable)
    return NumberFieldDiscriminantResult(discriminant=disc)
