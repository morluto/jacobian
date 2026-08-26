"""Domain-owned Diophantine approximation operations and claim verifiers."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.diophantine_approximation import (
    continued_fraction,
    convergents,
    solve_pell,
)
from jacobian.math.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ContinuedFractionResult,
    ConvergentRequest,
    ConvergentResult,
    ConvergentValue,
    PellEquationRequest,
    PellEquationResult,
)


def compute_continued_fraction(
    request: ContinuedFractionRequest,
) -> ContinuedFractionResult:
    coefficients, preperiod_length, period_length = continued_fraction(
        request.discriminant, request.term_count
    )
    return ContinuedFractionResult._from_kernel(
        discriminant=request.discriminant,
        term_count=request.term_count,
        coefficients=tuple(coefficients),
        preperiod_length=preperiod_length,
        period_length=period_length,
    )


def compute_convergents(request: ConvergentRequest) -> ConvergentResult:
    values = convergents(request.discriminant, request.convergent_count)
    return ConvergentResult._from_kernel(
        discriminant=request.discriminant,
        convergent_count=request.convergent_count,
        convergents=tuple(
            ConvergentValue(
                index=index,
                numerator=format_canonical_integer(numerator),
                denominator=format_canonical_integer(denominator),
            )
            for index, numerator, denominator in values
        ),
    )


def compute_pell_equation(request: PellEquationRequest) -> PellEquationResult:
    x, y = solve_pell(request.discriminant)
    return PellEquationResult._from_kernel(
        discriminant=request.discriminant,
        x=format_canonical_integer(x),
        y=format_canonical_integer(y),
    )


def verify_continued_fraction_result(result: ContinuedFractionResult) -> bool:
    """Verify an independently supplied continued-fraction claim.

    Result parsing is intentionally structural.  This bounded replay is the
    opt-in semantic check: ``term_count`` has already been limited by the
    request carrier before the SymPy-backed coefficient stream is reached.
    """

    try:
        coefficients, preperiod_length, period_length = continued_fraction(
            result.discriminant, result.term_count
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return (
        tuple(coefficients) == result.coefficients
        and preperiod_length == result.preperiod_length
        and period_length == result.period_length
    )


def verify_convergent_result(result: ConvergentResult) -> bool:
    """Verify an independently supplied convergent sequence in its envelope."""

    try:
        expected = convergents(result.discriminant, result.convergent_count)
    except (ArithmeticError, TypeError, ValueError):
        return False
    return all(
        value.index == index
        and parse_canonical_integer(value.numerator) == numerator
        and parse_canonical_integer(value.denominator) == denominator
        for value, (index, numerator, denominator) in zip(
            result.convergents, expected, strict=True
        )
    )


def verify_pell_equation_result(result: PellEquationResult) -> bool:
    """Verify that an independently supplied result is the fundamental solution."""

    try:
        return (
            parse_canonical_integer(result.x),
            parse_canonical_integer(result.y),
        ) == solve_pell(result.discriminant)
    except (ArithmeticError, TypeError, ValueError):
        return False


__all__ = [
    "compute_continued_fraction",
    "compute_convergents",
    "compute_pell_equation",
    "verify_continued_fraction_result",
    "verify_convergent_result",
    "verify_pell_equation_result",
]
