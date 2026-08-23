"""Domain-owned recurrence solving."""

from __future__ import annotations

from jacobian.math.recurrence_solving import (
    berlekamp_massey,
    closed_form,
    find_recurrence,
)
from jacobian.math.recurrence_solving._models import (
    ClosedFormRequest,
    ClosedFormResult,
    PrimeFieldRecurrenceFindRequest,
    PrimeFieldRecurrenceFindResult,
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


def compute_prime_field_find_recurrence(
    request: PrimeFieldRecurrenceFindRequest,
) -> PrimeFieldRecurrenceFindResult:
    """Find the minimal LFSR over ``GF(p)`` via Berlekamp-Massey.

    The recurrence is established only on the supplied prefix
    ``L <= n < len(sequence)``.
    """
    rec = berlekamp_massey(list(request.sequence), request.prime)
    return PrimeFieldRecurrenceFindResult(
        sequence=request.sequence,
        recurrence=rec,
    )
