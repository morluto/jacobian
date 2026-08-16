"""Domain adapter for number field operations."""
from __future__ import annotations
from jacobian.contracts.number_field import (
    NumberFieldRequest, NumberFieldDiscriminantResult, NumberFieldRingOfIntegersResult,
)
from jacobian.math.number_field import discriminant, ring_of_integers

def compute_nf_discriminant(request: NumberFieldRequest) -> NumberFieldDiscriminantResult:
    disc = discriminant(list(request.coefficients_descending), request.variable)
    return NumberFieldDiscriminantResult(discriminant=disc)

def compute_nf_ring_of_integers(request: NumberFieldRequest) -> NumberFieldRingOfIntegersResult:
    basis = ring_of_integers(list(request.coefficients_descending), request.variable)
    return NumberFieldRingOfIntegersResult(integral_basis=tuple(basis))
