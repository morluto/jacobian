"""Exact arithmetic-function operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.number_theory.arithmetic_functions import operations as native
from jacobian.math.number_theory.arithmetic_functions._models import (
    DirichletConvolutionRequest,
    DirichletConvolutionResult,
    DirichletInverseRequest,
    DirichletInverseResult,
    MobiusTransformRequest,
    MobiusTransformResult,
    SummatoryFunctionRequest,
    SummatoryFunctionResult,
)


def compute_dirichlet_convolution(
    request: DirichletConvolutionRequest,
) -> DirichletConvolutionResult:
    values = native.dirichlet_convolution(request.f, request.g)
    return DirichletConvolutionResult(values=values, length=len(values))


def compute_mobius_transform(
    request: MobiusTransformRequest,
) -> MobiusTransformResult:
    values = native.mobius_transform(request.values, request.inverse)
    return MobiusTransformResult(
        values=values, length=len(values), inverse=request.inverse
    )


def compute_summatory_function(
    request: SummatoryFunctionRequest,
) -> SummatoryFunctionResult:
    values = native.summatory_function(request.values)
    return SummatoryFunctionResult(values=values, length=len(values))


def compute_dirichlet_inverse(
    request: DirichletInverseRequest,
) -> DirichletInverseResult:
    values = native.dirichlet_inverse(request.values)
    return DirichletInverseResult(values=values, length=len(values))


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="arithmetic.dirichlet_convolution.compute",
        title="Compute the Dirichlet convolution of two arithmetic functions",
        description="Given two arithmetic functions f and g as tuples of values at indices 1, 2, ..., n, compute (f*g)(K) = sum_{d|K} f(d) * g(K/d) for K = 1..n using exact rational arithmetic.",
        request_type=DirichletConvolutionRequest,
        result_type=DirichletConvolutionResult,
        run=compute_dirichlet_convolution,
        tags=("arithmetic", "dirichlet-convolution", "exact"),
        examples=(
            OperationExample(
                name="identity_convolution",
                description="Dirichlet convolution of the identity function with the constant-one function gives the divisor-count function tau.",
                input={
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
    MathTool(
        operation_id="arithmetic.mobius_transform.compute",
        title="Compute the Möbius transform (or inverse) of an arithmetic function",
        description="Given an arithmetic function F at indices 1..n, compute the forward Möbius transform f(K) = sum_{d|K} mu(d) * F(K/d) (Dirichlet convolution with the Möbius function). When inverse is true, apply the inverse Möbius transform F(K) = sum_{d|K} f(K/d) (Dirichlet convolution with the constant-one function). The forward and inverse transforms are mutually inverse: forward then inverse (or vice versa) recovers the original function.",
        request_type=MobiusTransformRequest,
        result_type=MobiusTransformResult,
        run=compute_mobius_transform,
        tags=("arithmetic", "mobius-transform", "exact"),
        examples=(
            OperationExample(
                name="mobius_transform_of_constant_one",
                description="The Möbius transform of the constant-one function is the epsilon function (1 at 1, 0 elsewhere).",
                input={
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
    MathTool(
        operation_id="arithmetic.summatory_function.compute",
        title="Compute the summatory function of an arithmetic function",
        description="Given exact values of an arithmetic function at indices 1 through n, "
        "compute the partial sums S(K) = sum_{i=1}^K f(i) for K = 1 through n.",
        request_type=SummatoryFunctionRequest,
        result_type=SummatoryFunctionResult,
        run=compute_summatory_function,
        tags=("arithmetic", "summatory", "exact"),
        examples=(
            OperationExample(
                name="identity_summatory",
                description="The partial sums of 1, 2, 3, 4 are 1, 3, 6, 10.",
                input={
                    "values": [
                        {"num": "1", "den": "1"},
                        {"num": "2", "den": "1"},
                        {"num": "3", "den": "1"},
                        {"num": "4", "den": "1"},
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="arithmetic.dirichlet_inverse.compute",
        title="Compute the Dirichlet inverse of an arithmetic function",
        description="Given an arithmetic function f at indices 1..n with f(1) != 0, compute the Dirichlet inverse g such that f * g = epsilon (where epsilon(1) = 1 and epsilon(n) = 0 for n > 1). Uses exact rational arithmetic with the recursive definition g(1) = 1/f(1) and g(K) = -(1/f(1)) * sum_{d|K, d>1} f(d) * g(K/d).",
        request_type=DirichletInverseRequest,
        result_type=DirichletInverseResult,
        run=compute_dirichlet_inverse,
        tags=("arithmetic", "dirichlet-inverse", "exact"),
        examples=(
            OperationExample(
                name="inverse_of_constant_one",
                description="The Dirichlet inverse of the constant-one function is the Möbius function mu.",
                input={
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


__all__ = ["TOOLS"]
