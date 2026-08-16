"""Domain adapter for number field operations."""

from __future__ import annotations

from jacobian.contracts.number_field import (
    NumberFieldDiscriminantResult,
    NumberFieldRequest,
)
from jacobian.math.number_field import discriminant


def compute_nf_discriminant(
    request: NumberFieldRequest,
) -> NumberFieldDiscriminantResult:
    disc = discriminant(list(request.coefficients_descending), request.variable)
    return NumberFieldDiscriminantResult(discriminant=disc)
