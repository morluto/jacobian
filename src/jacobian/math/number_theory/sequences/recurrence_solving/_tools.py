"""Recurrence solving operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
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
    return ClosedFormResult._from_kernel(expression=result.expression)


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


def rs_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


FIBONACCI_MOD_7 = {"prime": 7, "sequence": [0, 1, 1, 2, 3, 5, 1, 6, 0]}

TOOLS: tuple[MathTool[Any, Any], ...] = (
    rs_operation(
        "sequence.recurrence.find",
        "Find the minimal linear recurrence of a sequence",
        "Find the lowest-order non-vacuous homogeneous recurrence that exactly fits the supplied finite rational sequence, or report NO_FITTING_RECURRENCE.",
        RecurrenceFindRequest,
        RecurrenceFindResult,
        compute_find_recurrence,
        "sequence",
        "recurrence",
        "exact",
        examples=(
            example(
                "fib_find",
                "Find the recurrence of the Fibonacci sequence.",
                {
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
    rs_operation(
        "sequence.recurrence.closed_form.compute",
        "Compute the closed-form of a linear recurrence",
        "Compute a SymPy-expression closed form for a characteristic polynomial of degree at most four and exactly one initial value per degree, including repeated roots.",
        ClosedFormRequest,
        ClosedFormResult,
        compute_closed_form,
        "sequence",
        "recurrence",
        "closed-form",
        "exact",
        examples=(
            example(
                "repeated_root",
                "Solve the recurrence with characteristic polynomial (x-1)^2.",
                {
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
    rs_operation(
        "sequence.recurrence.prime_field.find",
        "Find the minimal linear recurrence over a prime field",
        "Given a finite sequence over an explicitly supplied prime field "
        "GF(p), find the minimal linear recurrence (LFSR connection) it "
        "satisfies on the supplied prefix using the Berlekamp-Massey "
        "algorithm, returning the canonical PrimeFieldRecurrence value with "
        "its coefficients over GF(p). Every admitted finite sequence fits a "
        "recurrence of order at most len(sequence), so the result is always "
        "a fitted recurrence. Established for indices L <= n < "
        "len(sequence) only; no claim about unobserved terms.",
        PrimeFieldRecurrenceFindRequest,
        PrimeFieldRecurrenceFindResult,
        compute_prime_field_find_recurrence,
        "sequence",
        "recurrence",
        "berlekamp-massey",
        "prime-field",
        "exact",
        examples=(
            example(
                "fibonacci_mod_7",
                (
                    "Find the minimal recurrence of Fibonacci mod 7 "
                    "[0,1,1,2,3,5,1,6,0]; the result is s_n = s_{n-1} + "
                    "s_{n-2}, i.e. coefficients (1, 1). Values must be "
                    "residues modulo the prime."
                ),
                FIBONACCI_MOD_7,
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
