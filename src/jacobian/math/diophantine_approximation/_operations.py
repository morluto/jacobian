"""Domain-owned Diophantine approximation operations."""

from __future__ import annotations

from jacobian.canonical import format_canonical_integer
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
    return ContinuedFractionResult(
        discriminant=request.discriminant,
        coefficients=tuple(coefficients),
        preperiod_length=preperiod_length,
        period_length=period_length,
    )


def compute_convergents(request: ConvergentRequest) -> ConvergentResult:
    values = convergents(request.discriminant, request.convergent_count)
    return ConvergentResult(
        discriminant=request.discriminant,
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
    return PellEquationResult(
        discriminant=request.discriminant,
        x=format_canonical_integer(x),
        y=format_canonical_integer(y),
    )
