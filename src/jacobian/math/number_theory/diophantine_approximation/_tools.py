"""Diophantine approximation operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.diophantine_approximation import operations as native
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
    return native.continued_fraction(request.discriminant, request.term_count)


def compute_convergents(request: ConvergentRequest) -> ConvergentResult:
    return native.convergents(request.discriminant, request.convergent_count)


def compute_pell_equation(request: PellEquationRequest) -> PellEquationResult:
    return native.solve_pell(request.discriminant)


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="diophantine.continued_fraction.compute",
        title="Compute the continued fraction expansion of sqrt(D)",
        description="Compute the exact continued fraction [a_0; a_1, ...] of sqrt(D) "
        "for a squarefree positive integer D, using SymPy's exact "
        "continued_fraction_periodic.",
        request_type=ContinuedFractionRequest,
        result_type=ContinuedFractionResult,
        run=compute_continued_fraction,
        tags=("number-theory", "continued-fraction", "exact"),
        examples=(
            OperationExample(
                name="sqrt_2",
                description="Continued fraction of sqrt(2) is [1; 2, 2, 2, ...].",
                input={"discriminant": 2, "term_count": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="diophantine.convergents.compute",
        title="Compute convergents of sqrt(D)",
        description="Compute the first n convergents p_k/q_k of sqrt(D) using the exact "
        "continued-fraction recurrence.",
        request_type=ConvergentRequest,
        result_type=ConvergentResult,
        run=compute_convergents,
        tags=("number-theory", "convergents", "exact"),
        examples=(
            OperationExample(
                name="sqrt_2_convergents",
                description="Compute the first five convergents of sqrt(2).",
                input={"discriminant": 2, "convergent_count": 5},
            ),
        ),
    ),
    MathTool(
        operation_id="diophantine.pell_equation.solve",
        title="Solve the Pell equation x^2 - D*y^2 = 1",
        description="Find the fundamental solution (x, y) to the Pell equation "
        "x^2 - D*y^2 = 1 by iterating through continued fraction "
        "convergents of sqrt(D).",
        request_type=PellEquationRequest,
        result_type=PellEquationResult,
        run=compute_pell_equation,
        tags=("number-theory", "pell-equation", "exact"),
        examples=(
            OperationExample(
                name="pell_2",
                description="The fundamental solution to x^2 - 2*y^2 = 1 is (3, 2).",
                input={"discriminant": 2},
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
