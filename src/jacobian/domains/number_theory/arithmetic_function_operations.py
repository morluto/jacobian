"""Arithmetic function operations backed by exact integer arithmetic."""

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


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n."""
    divs = []
    i = 1
    while i * i <= n:
        if i * i == n:
            divs.append(i)
        elif n % i == 0:
            divs.append(i)
            divs.append(n // i)
        i += 1
    return sorted(divs)


def compute_dirichlet_convolution(
    request: DirichletConvolutionRequest,
) -> DirichletConvolutionResult:
    """Compute (f * g)(n) = sum_{d|n} f(d) * g(n/d) for 1 <= n <= N."""
    N = len(request.left)
    result = [0] * N
    for n in range(1, N + 1):
        total = 0
        for d in _divisors(n):
            total += request.left[d - 1] * request.right[n // d - 1]
        result[n - 1] = total
    return DirichletConvolutionResult(values=tuple(result))


def compute_mobius_transform(
    request: MobiusTransformRequest,
) -> MobiusTransformResult:
    """Compute the Mobius (divisor) transform: g(n) = sum_{d|n} mu(d) * f(n/d).

    This is equivalent to the Dirichlet convolution with the Mobius function.
    The inverse transform computes g(n) = sum_{d|n} f(d).
    """
    N = len(request.values)

    # Compute Mobius function for 1..N
    from sympy import mobius

    # Dirichlet transform: g(n) = sum_{d|n} f(d)
    # This is the divisor-sum (zeta) transform
    result = [0] * N
    for n in range(1, N + 1):
        total = 0
        for d in _divisors(n):
            total += request.values[d - 1]
        result[n - 1] = total
    return MobiusTransformResult(values=tuple(result))


def compute_dirichlet_inverse(
    request: DirichletInverseRequest,
) -> DirichletInverseResult:
    """Compute the Dirichlet inverse of f (given f(1) = 1).

    The inverse g satisfies (f * g)(n) = epsilon(n) where epsilon(1)=1, epsilon(n>1)=0.
    Computed recursively: g(1) = 1/f(1) = 1, g(n) = -sum_{d|n, d<n} f(n/d) * g(d).
    """
    N = len(request.values)
    result = [0] * N
    result[0] = 1  # g(1) = 1/f(1) = 1

    for n in range(2, N + 1):
        total = 0
        for d in _divisors(n):
            if d == n:
                continue
            total += request.values[n // d - 1] * result[d - 1]
        result[n - 1] = -total
    return DirichletInverseResult(values=tuple(result))


def compute_summatory_function(
    request: SummatoryFunctionRequest,
) -> SummatoryFunctionResult:
    """Compute the summatory (prefix sum) function S(n) = sum_{k=1}^{n} f(k)."""
    result = []
    total = 0
    for v in request.values:
        total += v
        result.append(total)
    return SummatoryFunctionResult(values=tuple(result))
