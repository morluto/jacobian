"""Domain adapter for Diophantine approximation operations."""

from __future__ import annotations

from jacobian.contracts.diophantine_approximation import (
    ContinuedFractionRequest,
    ContinuedFractionResult,
    ConvergentRequest,
    ConvergentResult,
    ConvergentValue,
    PellEquationRequest,
    PellEquationResult,
)
from jacobian.math.diophantine_approximation import (
    continued_fraction,
    convergents,
    solve_pell,
)


def compute_continued_fraction(
    request: ContinuedFractionRequest,
) -> ContinuedFractionResult:
    disc = request.discriminant
    count = request.term_count
    coeffs, preperiod_len, period_len = continued_fraction(disc, count)
    return ContinuedFractionResult(
        discriminant=disc,
        coefficients=tuple(coeffs),
        preperiod_length=preperiod_len,
        period_length=period_len,
    )


def compute_convergents(request: ConvergentRequest) -> ConvergentResult:
    disc = request.discriminant
    count = request.convergent_count
    convs = convergents(disc, count)
    return ConvergentResult(
        discriminant=disc,
        convergents=tuple(
            ConvergentValue(index=idx, numerator=p, denominator=q)
            for idx, p, q in convs
        ),
    )


def compute_pell_equation(request: PellEquationRequest) -> PellEquationResult:
    disc = request.discriminant
    x, y = solve_pell(disc)
    return PellEquationResult(discriminant=disc, x=x, y=y)
