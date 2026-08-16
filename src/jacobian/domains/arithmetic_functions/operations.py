"""Domain adapter for arithmetic-function operations."""

from __future__ import annotations

from fractions import Fraction

from jacobian.canonical import format_canonical_integer
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
from jacobian.contracts.exact import CanonicalRational


def _rational(value: Fraction | int) -> CanonicalRational:
    """Convert an exact Fraction or int to a CanonicalRational."""
    frac = Fraction(value)
    return CanonicalRational(
        num=format_canonical_integer(frac.numerator),
        den=format_canonical_integer(frac.denominator),
    )


def _divisors(n: int) -> list[int]:
    """Return the divisors of ``n`` in increasing order."""
    if n <= 0:
        return []
    small: list[int] = []
    large: list[int] = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def _mobius_sieve(n: int) -> list[int]:
    """Return ``mu`` for 1..n where ``mu`` is the Möbius function.

    ``mu(1) = 1``, and for ``k > 1`` ``mu(k) = 0`` if ``k`` has a squared
    prime factor, otherwise ``(-1)^(number of distinct prime factors)``.
    """
    mobius = [0] * (n + 1)
    if n >= 1:
        mobius[1] = 1
    is_prime = [True] * (n + 1)
    smallest_prime_factor = [0] * (n + 1)
    for i in range(2, n + 1):
        if is_prime[i]:
            smallest_prime_factor[i] = i
            for j in range(i * i, n + 1, i):
                if is_prime[j]:
                    is_prime[j] = False
                    smallest_prime_factor[j] = i
        p = smallest_prime_factor[i]
        if p == i:
            # i is prime
            mobius[i] = -1
        else:
            # i = i // p; check if p divides (i // p)
            reduced = i // p
            if reduced % p == 0:
                mobius[i] = 0
            else:
                mobius[i] = -mobius[reduced]
    return mobius


def compute_dirichlet_convolution(
    request: DirichletConvolutionRequest,
) -> DirichletConvolutionResult:
    """Compute ``h = f * g`` where ``h(K) = sum_{d|K} f(d) * g(K/d)``."""
    n = len(request.f)
    f = [v.as_fraction() for v in request.f]
    g = [v.as_fraction() for v in request.g]
    # Precompute divisors for each k = 1..n.
    result_values: list[Fraction] = [Fraction(0)] * n
    for k in range(1, n + 1):
        acc = Fraction(0)
        for d in _divisors(k):
            acc += f[d - 1] * g[k // d - 1]
        result_values[k - 1] = acc
    return DirichletConvolutionResult(
        values=tuple(_rational(v) for v in result_values),
        length=n,
    )


def compute_mobius_transform(
    request: MobiusTransformRequest,
) -> MobiusTransformResult:
    """Compute the Möbius (inverse) transform.

    Forward:  ``f(K) = sum_{d|K} mu(d) * F(K/d)`` (input is F, output is f).
    Inverse: ``F(K) = sum_{d|K} f(K/d)``           (input is f, output is F).

    The forward transform is Dirichlet convolution with the Möbius function
    ``mu`` (``f = mu * F``); the inverse is Dirichlet convolution with the
    constant-one function ``1`` (``F = 1 * f``), since ``mu * 1 = epsilon``.
    The two operations are mutually inverse: forward then inverse (or vice
    versa) recovers the original function.
    """
    n = len(request.values)
    values = [v.as_fraction() for v in request.values]
    result_values: list[Fraction] = [Fraction(0)] * n
    if request.inverse:
        # F(K) = sum_{d|K} f(K/d)  (Dirichlet convolution with 1)
        for k in range(1, n + 1):
            acc = Fraction(0)
            for d in _divisors(k):
                acc += values[k // d - 1]
            result_values[k - 1] = acc
    else:
        # f(K) = sum_{d|K} mu(d) * F(K/d)  (Dirichlet convolution with mu)
        mobius = _mobius_sieve(n)
        for k in range(1, n + 1):
            acc = Fraction(0)
            for d in _divisors(k):
                if mobius[d] == 0:
                    continue
                acc += mobius[d] * values[k // d - 1]
            result_values[k - 1] = acc
    return MobiusTransformResult(
        values=tuple(_rational(v) for v in result_values),
        length=n,
        inverse=request.inverse,
    )


def compute_summatory_function(
    request: SummatoryFunctionRequest,
) -> SummatoryFunctionResult:
    """Compute ``S(K) = sum_{i=1}^{K} f(i)`` for K = 1..n."""
    n = len(request.values)
    values = [v.as_fraction() for v in request.values]
    result_values: list[Fraction] = [Fraction(0)] * n
    running = Fraction(0)
    for i in range(n):
        running += values[i]
        result_values[i] = running
    return SummatoryFunctionResult(
        values=tuple(_rational(v) for v in result_values),
        length=n,
    )


def compute_dirichlet_inverse(
    request: DirichletInverseRequest,
) -> DirichletInverseResult:
    """Compute the Dirichlet inverse ``g`` of ``f`` such that ``f * g = epsilon``.

    The Dirichlet inverse is defined recursively:
    ``g(1) = 1 / f(1)`` and for ``K > 1``:
    ``g(K) = -(1 / f(1)) * sum_{d | K, d > 1} f(d) * g(K / d)``.

    The first element of ``f`` (i.e. ``f(1)``) must be non-zero.
    """
    f = [v.as_fraction() for v in request.values]
    n = len(request.values)
    g: list[Fraction] = [Fraction(0)] * n
    g[0] = Fraction(1) / f[0]
    for k in range(2, n + 1):
        divs = _divisors(k)
        partial = Fraction(0)
        for d in divs:
            if d == 1:
                continue
            partial += f[d - 1] * g[k // d - 1]
        g[k - 1] = -(partial / f[0])
    return DirichletInverseResult(
        values=tuple(_rational(v) for v in g),
        length=n,
    )
