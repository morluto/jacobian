"""Exact arithmetic-function operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletConvolutionResult,
    DirichletInverseRequest,
    DirichletInverseResult,
    MobiusTransformRequest,
    MobiusTransformResult,
    SummatoryFunctionRequest,
    SummatoryFunctionResult,
)
from jacobian.math.arithmetic_functions._operations import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)


def arithmetic_functions_operation[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
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


ARITHMETIC_FUNCTIONS_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    arithmetic_functions_operation(
        "arithmetic.dirichlet_convolution.compute",
        "Compute the Dirichlet convolution of two arithmetic functions",
        "Given two arithmetic functions f and g as tuples of values at indices 1, 2, ..., n, compute (f*g)(K) = sum_{d|K} f(d) * g(K/d) for K = 1..n using exact rational arithmetic.",
        DirichletConvolutionRequest,
        DirichletConvolutionResult,
        compute_dirichlet_convolution,
        "arithmetic",
        "dirichlet-convolution",
        "exact",
        examples=(
            example(
                "identity_convolution",
                "Dirichlet convolution of the identity function with the constant-one function gives the divisor-count function tau.",
                {
                    "f": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "4", "den": "1"},
                    ],
                    "g": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    arithmetic_functions_operation(
        "arithmetic.mobius_transform.compute",
        "Compute the Möbius transform (or inverse) of an arithmetic function",
        "Given an arithmetic function F at indices 1..n, compute the forward Möbius transform f(K) = sum_{d|K} mu(d) * F(K/d) (Dirichlet convolution with the Möbius function). When inverse is true, apply the inverse Möbius transform F(K) = sum_{d|K} f(K/d) (Dirichlet convolution with the constant-one function). The forward and inverse transforms are mutually inverse: forward then inverse (or vice versa) recovers the original function.",
        MobiusTransformRequest,
        MobiusTransformResult,
        compute_mobius_transform,
        "arithmetic",
        "mobius-transform",
        "exact",
        examples=(
            example(
                "mobius_transform_of_constant_one",
                "The Möbius transform of the constant-one function is the epsilon function (1 at 1, 0 elsewhere).",
                {
                    "values": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                    "inverse": False,
                },
            ),
        ),
    ),
    arithmetic_functions_operation(
        "arithmetic.summatory_function.compute",
        "Compute the summatory function (partial sums) of an arithmetic function",
        "Given an arithmetic function f at indices 1..n, compute S(K) = sum_{i=1}^{K} f(i) for K = 1..n using exact rational arithmetic.",
        SummatoryFunctionRequest,
        SummatoryFunctionResult,
        compute_summatory_function,
        "arithmetic",
        "summatory",
        "exact",
        examples=(
            example(
                "identity_summatory",
                "Summatory function of the identity function gives the triangular numbers.",
                {
                    "values": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "4", "den": "1"},
                    ],
                },
            ),
        ),
    ),
    arithmetic_functions_operation(
        "arithmetic.dirichlet_inverse.compute",
        "Compute the Dirichlet inverse of an arithmetic function",
        "Given an arithmetic function f at indices 1..n with f(1) != 0, compute the Dirichlet inverse g such that f * g = epsilon (where epsilon(1) = 1 and epsilon(n) = 0 for n > 1). Uses exact rational arithmetic with the recursive definition g(1) = 1/f(1) and g(K) = -(1/f(1)) * sum_{d|K, d>1} f(d) * g(K/d).",
        DirichletInverseRequest,
        DirichletInverseResult,
        compute_dirichlet_inverse,
        "arithmetic",
        "dirichlet-inverse",
        "exact",
        examples=(
            example(
                "inverse_of_constant_one",
                "The Dirichlet inverse of the constant-one function is the Möbius function mu.",
                {
                    "values": [
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                        {"num": "1", "den": "1"},
                    ],
                },
            ),
        ),
    ),
)

TOOLS = ARITHMETIC_FUNCTIONS_OPERATIONS

__all__ = ["TOOLS"]
