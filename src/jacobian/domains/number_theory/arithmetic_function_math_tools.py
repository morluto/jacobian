"""MathTool declarations for arithmetic function operations."""

from __future__ import annotations

from jacobian.contracts.arithmetic_functions import (
    DirichletConvolutionRequest,
    DirichletConvolutionResult,
    DirichletInverseRequest,
    DirichletInverseResult,
    MobiusTransformRequest,
    MobiusTransformResult,
    SummatoryFunctionRequest,
    SummatoryFunctionResult,
)
from jacobian.domains._examples import example
from jacobian.domains.number_theory.arithmetic_function_operations import (
    compute_dirichlet_convolution,
    compute_dirichlet_inverse,
    compute_mobius_transform,
    compute_summatory_function,
)
from jacobian.math_tools import MathTool


ARITHMETIC_FUNCTION_OPERATIONS: tuple[MathTool, ...] = (
    MathTool(
        operation_id="number_theory.arithmetic_function.dirichlet_convolution.compute",
        version="1",
        title="Compute the Dirichlet convolution of two arithmetic functions",
        description=(
            "Compute (f * g)(n) = sum_{d|n} f(d) * g(n/d) for 1 <= n <= N "
            "from two bounded tables f and g."
        ),
        request_type=DirichletConvolutionRequest,
        result_type=DirichletConvolutionResult,
        run=compute_dirichlet_convolution,
        tags=(
            "number-theory",
            "arithmetic-function",
            "dirichlet",
            "convolution",
            "exact",
        ),
        examples=(
            example(
                "id_convolution",
                "Dirichlet convolution of the identity with itself.",
                {
                    "left": [1, 2, 3, 4],
                    "right": [1, 2, 3, 4],
                },
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.arithmetic_function.divisor_transform.compute",
        version="1",
        title="Compute the divisor (zeta) transform of an arithmetic function",
        description=(
            "Compute g(n) = sum_{d|n} f(d) for 1 <= n <= N, the divisor-sum "
            "(zeta) transform of a bounded arithmetic function table."
        ),
        request_type=MobiusTransformRequest,
        result_type=MobiusTransformResult,
        run=compute_mobius_transform,
        tags=(
            "number-theory",
            "arithmetic-function",
            "divisor-transform",
            "zeta",
            "exact",
        ),
        examples=(
            example(
                "divisor_sum",
                "Divisor sum of [1,1,1,1] is [1,3,1,7].",
                {"values": [1, 1, 1, 1]},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.arithmetic_function.dirichlet_inverse.compute",
        version="1",
        title="Compute the Dirichlet inverse of an arithmetic function",
        description=(
            "Compute the Dirichlet inverse g such that (f * g)(n) = "
            "epsilon(n), requiring f(1) = 1."
        ),
        request_type=DirichletInverseRequest,
        result_type=DirichletInverseResult,
        run=compute_dirichlet_inverse,
        tags=(
            "number-theory",
            "arithmetic-function",
            "dirichlet",
            "inverse",
            "exact",
        ),
        examples=(
            example(
                "identity_inverse",
                "Dirichlet inverse of the identity function.",
                {"values": [1, 2, 3, 4, 5]},
            ),
        ),
    ),
    MathTool(
        operation_id="number_theory.arithmetic_function.summatory.compute",
        version="1",
        title="Compute the summatory (prefix sum) of an arithmetic function",
        description=(
            "Compute S(n) = sum_{k=1}^{n} f(k) for 1 <= n <= N, the "
            "prefix sum of a bounded arithmetic function table."
        ),
        request_type=SummatoryFunctionRequest,
        result_type=SummatoryFunctionResult,
        run=compute_summatory_function,
        tags=(
            "number-theory",
            "arithmetic-function",
            "summatory",
            "prefix-sum",
            "exact",
        ),
        examples=(
            example(
                "prefix_sum",
                "Summatory function of [1,2,3,4] is [1,3,6,10].",
                {"values": [1, 2, 3, 4]},
            ),
        ),
    ),
)

__all__ = ["ARITHMETIC_FUNCTION_OPERATIONS"]
