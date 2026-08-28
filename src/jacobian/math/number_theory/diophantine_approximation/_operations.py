"""Domain-owned Diophantine approximation operations."""

from __future__ import annotations

from jacobian.math.number_theory.diophantine_approximation import (
    continued_fraction,
    convergents,
    solve_pell,
)
from jacobian.math.number_theory.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ContinuedFractionResult,
    ConvergentRequest,
    ConvergentResult,
    PellEquationRequest,
    PellEquationResult,
)


def compute_continued_fraction(
    request: ContinuedFractionRequest,
) -> ContinuedFractionResult:
    return continued_fraction(request.discriminant, request.term_count)


def compute_convergents(request: ConvergentRequest) -> ConvergentResult:
    return convergents(request.discriminant, request.convergent_count)


def compute_pell_equation(request: PellEquationRequest) -> PellEquationResult:
    return solve_pell(request.discriminant)


__all__ = [
    "compute_continued_fraction",
    "compute_convergents",
    "compute_pell_equation",
]
