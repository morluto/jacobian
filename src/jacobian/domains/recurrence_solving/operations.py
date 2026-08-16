"""Domain adapter for recurrence solving."""
from __future__ import annotations
from jacobian.contracts.recurrence_solving import (
    ClosedFormRequest, ClosedFormResult, RecurrenceFindRequest, RecurrenceFindResult,
)
from jacobian.math.recurrence_solving import find_recurrence, closed_form

def compute_find_recurrence(request: RecurrenceFindRequest) -> RecurrenceFindResult:
    result = find_recurrence(list(request.sequence))
    return RecurrenceFindResult(
        coefficients=result["coefficients"],
        order=result["order"],
    )

def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = closed_form(
        list(request.character_coefficients),
        list(request.initial_values),
    )
    return ClosedFormResult(expression=result["expression"])
