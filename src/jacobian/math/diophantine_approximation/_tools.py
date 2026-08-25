"""Diophantine approximation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.diophantine_approximation._models import (
    ContinuedFractionRequest,
    ContinuedFractionResult,
    ConvergentRequest,
    ConvergentResult,
    PellEquationRequest,
    PellEquationResult,
)
from jacobian.math.diophantine_approximation._operations import (
    compute_continued_fraction,
    compute_convergents,
    compute_pell_equation,
)


def da_operation[RequestT: StrictModel, ResultT: StrictModel](
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


DIOPHANTINE_APPROXIMATION_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    da_operation(
        "diophantine.continued_fraction.compute",
        "Compute the continued fraction expansion of sqrt(D)",
        "Compute the exact continued fraction [a_0; a_1, ...] of sqrt(D) "
        "for a squarefree positive integer D, using SymPy's exact "
        "continued_fraction_periodic.",
        ContinuedFractionRequest,
        ContinuedFractionResult,
        compute_continued_fraction,
        "number-theory",
        "continued-fraction",
        "exact",
        examples=(
            example(
                "sqrt_2",
                "Continued fraction of sqrt(2) is [1; 2, 2, 2, ...].",
                {"discriminant": 2, "term_count": 5},
            ),
        ),
    ),
    da_operation(
        "diophantine.convergents.compute",
        "Compute convergents of sqrt(D)",
        "Compute the first n convergents p_n/q_n of sqrt(D) by the "
        "standard recurrence from the continued fraction expansion.",
        ConvergentRequest,
        ConvergentResult,
        compute_convergents,
        "number-theory",
        "convergents",
        "exact",
        examples=(
            example(
                "sqrt_2_convergents",
                "First 5 convergents of sqrt(2): 1/1, 3/2, 7/5, 17/12, 41/29.",
                {"discriminant": 2, "convergent_count": 5},
            ),
        ),
    ),
    da_operation(
        "diophantine.pell_equation.solve",
        "Solve the Pell equation x^2 - D*y^2 = 1",
        "Find the fundamental solution (x, y) to the Pell equation "
        "x^2 - D*y^2 = 1 by iterating through continued fraction "
        "convergents of sqrt(D).",
        PellEquationRequest,
        PellEquationResult,
        compute_pell_equation,
        "number-theory",
        "pell-equation",
        "exact",
        examples=(
            example(
                "pell_2",
                "The fundamental solution to x^2 - 2*y^2 = 1 is (3, 2).",
                {"discriminant": 2},
            ),
        ),
    ),
)

TOOLS = DIOPHANTINE_APPROXIMATION_OPERATIONS

__all__ = ["TOOLS"]
