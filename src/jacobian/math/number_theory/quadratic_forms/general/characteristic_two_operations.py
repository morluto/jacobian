"""Admitted polar pairing for characteristic-two quadratic forms."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._flint import context, coordinates, to_backend
from jacobian.math.finite_fields.values import FiniteFieldElement
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two import (
    FiniteFieldQuadraticForm,
    FiniteFieldQuadraticPairingRequest,
    FiniteFieldQuadraticPairingResult,
    FiniteFieldQuadraticVector,
)

__all__ = ["polar_pairing_finite_field_quadratic_form"]


def _active_context(presentation: Any) -> Any:
    if presentation.characteristic != 2:
        raise OperationDomainValidationError(
            location=("form", "field", "characteristic"),
            code="quadratic_form.characteristic_two.requires_characteristic_two",
            message="this operation requires a field of characteristic two",
        )
    # `context` establishes primality, irreducibility, and exact power-basis
    # preservation once, before any form arithmetic.
    return context(presentation)


def polar_pairing_finite_field_quadratic_form(
    form: FiniteFieldQuadraticForm,
    left: FiniteFieldQuadraticVector,
    right: FiniteFieldQuadraticVector,
) -> FiniteFieldQuadraticPairingResult:
    """Compute Q(x+y)-Q(x)-Q(y); square terms cancel in characteristic two.

    A caller-supplied native value is re-admitted through the request schema
    before use. Only the matching source triple reaches the backend.
    """

    try:
        request = FiniteFieldQuadraticPairingRequest(form=form, left=left, right=right)
    except Exception as error:
        raise OperationDomainValidationError(
            location=("form", "left", "right"),
            code="quadratic_form.characteristic_two.invalid_request",
            message="pairing inputs violate finite-field quadratic invariants",
        ) from error
    if len(request.form.axis) + len(request.form.cross_terms) > 4096:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.characteristic_two.support_bound",
            message="quadratic-form support exceeds the admitted traversal bound",
        )
    active_context = _active_context(request.form.field)
    left_values = tuple(
        to_backend(value, active_context=active_context)
        for value in request.left.coordinates
    )
    right_values = tuple(
        to_backend(value, active_context=active_context)
        for value in request.right.coordinates
    )
    result = active_context(0)
    for term in request.form.cross_terms:
        coefficient = to_backend(term.coefficient, active_context=active_context)
        result += coefficient * (
            left_values[term.left] * right_values[term.right]
            + left_values[term.right] * right_values[term.left]
        )
    return FiniteFieldQuadraticPairingResult._from_kernel(
        form=request.form,
        left=request.left,
        right=request.right,
        value=FiniteFieldElement(
            presentation=request.form.field,
            coordinates=coordinates(result, degree=request.form.field.degree),
        ),
    )
