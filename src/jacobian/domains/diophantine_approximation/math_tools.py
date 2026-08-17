"""Diophantine approximation operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.diophantine_approximation import (
    ContinuedFractionRequest,
    ContinuedFractionResult,
    ConvergentRequest,
    ConvergentResult,
    PellEquationRequest,
    PellEquationResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.diophantine_approximation.operations import (
    compute_continued_fraction,
    compute_convergents,
    compute_pell_equation,
)
from jacobian.math_tools import MathTool


def da_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
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
