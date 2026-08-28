"""Domain-owned recurrence solving."""

from __future__ import annotations

from collections.abc import Callable

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory.sequences.recurrence_solving import (
    berlekamp_massey,
    closed_form,
    find_recurrence,
)
from jacobian.math.number_theory.sequences.recurrence_solving._models import (
    ClosedFormRequest,
    ClosedFormResult,
    PrimeFieldRecurrenceFindRequest,
    PrimeFieldRecurrenceFindResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)


def _run_admitted[T](operation: Callable[[], T], *, location: tuple[str, ...]) -> T:
    try:
        return operation()
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=location,
            code="recurrence_solving.invalid_domain",
            message=str(exc),
        ) from exc


def compute_find_recurrence(request: RecurrenceFindRequest) -> RecurrenceFindResult:
    result = _run_admitted(
        lambda: find_recurrence(request.sequence), location=("sequence",)
    )
    return RecurrenceFindResult._from_kernel(
        coefficients=result.coefficients,
        order=result.order,
        status=result.status,
    )


def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = _run_admitted(
        lambda: closed_form(
            request.characteristic_coefficients,
            request.initial_values,
        ),
        location=("characteristic_coefficients", "initial_values"),
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
