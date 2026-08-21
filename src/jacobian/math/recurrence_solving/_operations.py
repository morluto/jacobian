"""Domain-owned recurrence solving."""

from __future__ import annotations

from jacobian.math.recurrence_solving import closed_form, find_recurrence
from jacobian.math.recurrence_solving._models import (
    ClosedFormRequest,
    ClosedFormResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)


def compute_find_recurrence(request: RecurrenceFindRequest) -> RecurrenceFindResult:
    result = find_recurrence(request.sequence)
    return RecurrenceFindResult(
        coefficients=result.coefficients,
        order=result.order,
        status=result.status,
    )


def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = closed_form(
        request.characteristic_coefficients,
        request.initial_values,
    )
    return ClosedFormResult(expression=result.expression)
