"""Domain adapter for recurrence solving."""

from __future__ import annotations

from jacobian.contracts.recurrence_solving import (
    ClosedFormRequest,
    ClosedFormResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)
from jacobian.math.recurrence_solving import closed_form, find_recurrence


def compute_find_recurrence(request: RecurrenceFindRequest) -> RecurrenceFindResult:
    result = find_recurrence(list(request.sequence))  # type: ignore[no-untyped-call]
    return RecurrenceFindResult(
        coefficients=result["coefficients"],
        order=result["order"],
    )


def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = closed_form(  # type: ignore[no-untyped-call]
        list(request.characteristic_coefficients),
        list(request.initial_values),
    )
    return ClosedFormResult(expression=result["expression"])
