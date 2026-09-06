"""Recurrence solving operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math.number_theory.sequences.recurrence_solving import (
    operations as native,
)
from jacobian.math.number_theory.sequences.recurrence_solving._models import (
    ClosedFormRequest,
    ClosedFormResult,
    PrimeFieldRecurrenceFindRequest,
    PrimeFieldRecurrenceFindResult,
    RecurrenceFindRequest,
    RecurrenceFindResult,
)


def _run_admitted[ResultT](
    operation: Callable[[], ResultT], *, location: tuple[str, ...]
) -> ResultT:
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
        lambda: native.find_recurrence(request.sequence), location=("sequence",)
    )
    return RecurrenceFindResult._from_kernel(
        sequence=request.sequence,
        coefficients=result.coefficients,
        order=result.order,
        status=result.status,
    )


def compute_closed_form(request: ClosedFormRequest) -> ClosedFormResult:
    result = _run_admitted(
        lambda: native.closed_form(
            request.characteristic_coefficients,
            request.initial_values,
        ),
        location=("characteristic_coefficients", "initial_values"),
    )
    return ClosedFormResult._from_kernel(
        characteristic_coefficients=request.characteristic_coefficients,
        initial_values=request.initial_values,
        expression=result.expression,
    )


def compute_prime_field_find_recurrence(
    request: PrimeFieldRecurrenceFindRequest,
) -> PrimeFieldRecurrenceFindResult:
    """Find the minimal LFSR over ``GF(p)`` via Berlekamp-Massey."""
    rec = _run_admitted(
        lambda: native.berlekamp_massey(list(request.sequence), request.prime),
        location=("prime", "sequence"),
    )
    return PrimeFieldRecurrenceFindResult._from_kernel(
        sequence=request.sequence,
        recurrence=rec,
    )


FIBONACCI_MOD_7 = {"prime": 7, "sequence": [0, 1, 1, 2, 3, 5, 1, 6, 0]}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="sequence.recurrence.find",
        title="Find the minimal linear recurrence of a sequence",
        description="Find the lowest-order non-vacuous homogeneous recurrence that exactly fits the supplied finite rational sequence, or report NO_FITTING_RECURRENCE.",
        request_type=RecurrenceFindRequest,
        result_type=RecurrenceFindResult,
        run=compute_find_recurrence,
        tags=("sequence", "recurrence", "exact"),
        examples=(
            OperationExample(
                name="fib_find",
                description="Find the recurrence of the Fibonacci sequence.",
                input={
                    "sequence": [
                        {"num": value, "den": "1"}
                        for value in (
                            "1",
                            "1",
                            "2",
                            "3",
                            "5",
                            "8",
                            "13",
                            "21",
                            "34",
                            "55",
                        )
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.recurrence.closed_form.compute",
        title="Compute the closed-form of a linear recurrence",
        description="Compute a SymPy-expression closed form for a characteristic polynomial of degree at most four and exactly one initial value per degree, including repeated roots.",
        request_type=ClosedFormRequest,
        result_type=ClosedFormResult,
        run=compute_closed_form,
        tags=("sequence", "recurrence", "closed-form", "exact"),
        examples=(
            OperationExample(
                name="repeated_root",
                description="Solve the recurrence with characteristic polynomial (x-1)^2.",
                input={
                    "characteristic_coefficients": [
                        {"num": "1", "den": "1"},
                        {"num": "-2", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "initial_values": [
                        {"num": "2", "den": "1"},
                        {"num": "5", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="sequence.recurrence.prime_field.find",
        title="Find the minimal linear recurrence over a prime field",
        description="Given a finite sequence over an explicitly supplied prime field "
        "GF(p), find the minimal linear recurrence (LFSR connection) it "
        "satisfies on the supplied prefix using the Berlekamp-Massey "
        "algorithm, returning the canonical PrimeFieldRecurrence value with "
        "its coefficients over GF(p). Every admitted finite sequence fits a "
        "recurrence of order at most len(sequence), so the result is always "
        "a fitted recurrence. Established for indices L <= n < "
        "len(sequence) only; no claim about unobserved terms.",
        request_type=PrimeFieldRecurrenceFindRequest,
        result_type=PrimeFieldRecurrenceFindResult,
        run=compute_prime_field_find_recurrence,
        tags=("sequence", "recurrence", "berlekamp-massey", "prime-field", "exact"),
        examples=(
            OperationExample(
                name="fibonacci_mod_7",
                description=(
                    "Find the minimal recurrence of Fibonacci mod 7 "
                    "[0,1,1,2,3,5,1,6,0]; the result is s_n = s_{n-1} + "
                    "s_{n-2}, i.e. coefficients (1, 1). Values must be "
                    "residues modulo the prime."
                ),
                input=FIBONACCI_MOD_7,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
