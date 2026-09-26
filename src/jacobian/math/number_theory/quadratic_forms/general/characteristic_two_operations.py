"""Admitted evaluation and polar pairing for characteristic-two forms."""

from __future__ import annotations

from typing import Any

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.finite_fields._flint import context, coordinates, to_backend
from jacobian.math.finite_fields.values import FiniteFieldElement
from jacobian.math.number_theory.quadratic_forms.general.characteristic_two import (
    FiniteFieldQuadraticEvaluationRequest,
    FiniteFieldQuadraticEvaluationResult,
    FiniteFieldQuadraticPairingRequest,
    FiniteFieldQuadraticPairingResult,
)


def _admit_characteristic_two(presentation: Any) -> Any:
    if presentation.characteristic != 2:
        raise OperationDomainValidationError(
            location=("form", "field", "characteristic"),
            code="quadratic_form.characteristic_two.requires_characteristic_two",
            message="this operation requires a field of characteristic two",
        )
    # `context` establishes primality, irreducibility, and exact power-basis
    # preservation once, before any form arithmetic.
    return context(presentation)


def evaluate_finite_field_quadratic_form(
    request: FiniteFieldQuadraticEvaluationRequest,
) -> FiniteFieldQuadraticEvaluationResult:
    """Evaluate the retained polynomial, including its square terms."""

    if not isinstance(request, FiniteFieldQuadraticEvaluationRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="quadratic_form.characteristic_two.evaluation_request_type",
            message="request must be a finite-field quadratic evaluation request",
        )
    try:
        request = FiniteFieldQuadraticEvaluationRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="quadratic_form.characteristic_two.invalid_request",
            message="request violates finite-field quadratic evaluation invariants",
        ) from exc
    form = request.form
    n = len(form.axis)
    if n + len(form.cross_terms) > 4096:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.characteristic_two.support_bound",
            message="quadratic-form support exceeds the admitted traversal bound",
        )
    active_context = _admit_characteristic_two(form.field)
    values = tuple(
        to_backend(value, active_context=active_context)
        for value in request.vector.coordinates
    )
    result = active_context(0)
    for coefficient, value in zip(form.diagonal_coefficients, values, strict=True):
        if not coefficient.is_zero and value:
            result += (
                to_backend(coefficient, active_context=active_context) * value * value
            )
    for term in form.cross_terms:
        left, right = values[term.left], values[term.right]
        if left and right:
            result += (
                to_backend(term.coefficient, active_context=active_context)
                * left
                * right
            )
    return FiniteFieldQuadraticEvaluationResult(
        form=form,
        vector=request.vector,
        value=FiniteFieldElement(
            presentation=form.field,
            coordinates=coordinates(result, degree=form.field.degree),
        ),
    )


def polar_pairing_finite_field_quadratic_form(
    request: FiniteFieldQuadraticPairingRequest,
) -> FiniteFieldQuadraticPairingResult:
    """Compute Q(x+y)-Q(x)-Q(y); square terms cancel in characteristic two."""

    if not isinstance(request, FiniteFieldQuadraticPairingRequest):
        raise OperationDomainValidationError(
            location=("request",),
            code="quadratic_form.characteristic_two.pairing_request_type",
            message="request must be a finite-field quadratic pairing request",
        )
    try:
        request = FiniteFieldQuadraticPairingRequest.model_validate(
            request.model_dump(mode="python")
        )
    except Exception as exc:
        raise OperationDomainValidationError(
            location=("request",),
            code="quadratic_form.characteristic_two.invalid_request",
            message="request violates finite-field quadratic pairing invariants",
        ) from exc
    form = request.form
    if len(form.axis) + len(form.cross_terms) > 4096:
        raise OperationDomainValidationError(
            location=("form",),
            code="quadratic_form.characteristic_two.support_bound",
            message="quadratic-form support exceeds the admitted traversal bound",
        )
    active_context = _admit_characteristic_two(form.field)
    left = tuple(
        to_backend(value, active_context=active_context)
        for value in request.left.coordinates
    )
    right = tuple(
        to_backend(value, active_context=active_context)
        for value in request.right.coordinates
    )
    result = active_context(0)
    for term in form.cross_terms:
        coefficient = to_backend(term.coefficient, active_context=active_context)
        result += coefficient * (
            left[term.left] * right[term.right] + left[term.right] * right[term.left]
        )
    return FiniteFieldQuadraticPairingResult(
        form=form,
        left=request.left,
        right=request.right,
        value=FiniteFieldElement(
            presentation=form.field,
            coordinates=coordinates(result, degree=form.field.degree),
        ),
    )
