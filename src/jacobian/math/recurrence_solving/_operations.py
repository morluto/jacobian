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
    return RecurrenceFindResult._from_kernel(
        coefficients=result.coefficients,
        order=result.order,
        status=result.status,
    )


def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = closed_form(
        request.characteristic_coefficients,
        request.initial_values,
    )
    return ClosedFormResult._from_kernel(expression=result.expression)


def compute_prime_field_find_recurrence(
    request: PrimeFieldRecurrenceFindRequest,
) -> PrimeFieldRecurrenceFindResult:
    """Find the minimal LFSR over ``GF(p)`` via Berlekamp-Massey.

    The recurrence is established only on the supplied prefix
    ``L <= n < len(sequence)``.
    """
    rec = berlekamp_massey(list(request.sequence), request.prime)
    return PrimeFieldRecurrenceFindResult._from_kernel(
        sequence=request.sequence,
        recurrence=rec,
    )


def verify_prime_field_recurrence_find_result(
    result: PrimeFieldRecurrenceFindResult,
) -> bool:
    """Verify an independently supplied Berlekamp-Massey result.

    The result's request bounds cap the replay at 256 terms and a prime below
    10,000.  This deliberately lives outside Pydantic validation: parsing an
    untrusted wire result must not execute the operation or import the native
    API.
    """

    recurrence = result.recurrence
    try:
        expected = berlekamp_massey(list(result.sequence), recurrence.prime)
    except ValueError:
        return False
    return recurrence == expected
