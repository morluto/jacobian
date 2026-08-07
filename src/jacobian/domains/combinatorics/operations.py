"""Exact combinatorics operations backed by maintained SymPy and stdlib APIs."""

from __future__ import annotations

from functools import reduce
from operator import mul

from jacobian.canonical import format_canonical_integer
from jacobian.contracts.combinatorics import (
    FibonacciPairRequest,
    FibonacciPairResult,
    IntegerListRequest,
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
    RationalResult,
)
from jacobian.contracts.exact import CanonicalRational

__all__ = [
    "bell",
    "bernoulli",
    "binomial",
    "catalan",
    "central_binomial",
    "compositions",
    "derangements",
    "double_factorial",
    "enumerate_integer_partitions",
    "factorial",
    "fibonacci",
    "fibonacci_pair",
    "lucas",
    "motzkin",
    "multinomial",
    "partition_number",
    "permutations",
    "stirling_first",
    "stirling_second",
]


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=str(int(value)))


def factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    import math

    n = request.n
    return _integer_result(math.factorial(n))


def double_factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.factorial2(n))


def derangements(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.subfactorial(n))


def binomial(request: NonnegativePairRequest) -> IntegerResult:
    import math

    pair = request
    if pair.k > pair.n:
        return _integer_result(0)
    return _integer_result(math.comb(pair.n, pair.k))


def multinomial(request: IntegerListRequest) -> IntegerResult:
    import math

    values = [int(v) for v in request.values]
    numerator = math.factorial(sum(values))
    denominator = reduce(mul, (math.factorial(v) for v in values), 1)
    return _integer_result(numerator // denominator)


def permutations(request: NonnegativePairRequest) -> IntegerResult:
    import math

    pair = request
    if pair.k > pair.n:
        return _integer_result(0)
    return _integer_result(math.perm(pair.n, pair.k))


def stirling_first(request: NonnegativePairRequest) -> IntegerResult:
    from sympy.functions.combinatorial.numbers import stirling

    pair = request
    return _integer_result(stirling(pair.n, pair.k, kind=1))


def stirling_second(request: NonnegativePairRequest) -> IntegerResult:
    from sympy.functions.combinatorial.numbers import stirling

    pair = request
    return _integer_result(stirling(pair.n, pair.k, kind=2))


def bell(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.bell(n))


def catalan(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.catalan(n))


def partition_number(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.partition(n))


def enumerate_integer_partitions(
    request: IntegerPartitionEnumerationRequest,
) -> IntegerPartitionEnumerationResult:
    """Enumerate all bounded partitions using ``sympy.utilities.partitions``."""
    from sympy.utilities.iterables import partitions

    value = request
    expanded_partitions: list[tuple[int, ...]] = []
    for multiplicities in partitions(value.n, m=value.max_parts):
        expanded_partitions.append(
            tuple(
                part
                for part in sorted(multiplicities, reverse=True)
                for _ in range(int(multiplicities[part]))
            )
        )
    return IntegerPartitionEnumerationResult(
        n=value.n,
        max_parts=value.max_parts,
        partitions=tuple(expanded_partitions),
    )


def fibonacci(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.fibonacci(n))


def fibonacci_pair(request: FibonacciPairRequest) -> FibonacciPairResult:
    """Compute two consecutive Fibonacci values."""
    import sympy

    n = request.n
    return FibonacciPairResult(
        n=n,
        f_n=str(sympy.fibonacci(n)),
        f_n_plus_one=str(sympy.fibonacci(n + 1)),
    )


def lucas(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.lucas(n))


def motzkin(request: NonnegativeIntegerRequest) -> IntegerResult:
    import sympy

    n = request.n
    return _integer_result(sympy.motzkin(n))


def bernoulli(request: NonnegativeIntegerRequest) -> RationalResult:
    import sympy

    n = request.n
    value = sympy.bernoulli(n)
    return RationalResult(
        value=CanonicalRational(
            num=format_canonical_integer(int(value.p)),
            den=format_canonical_integer(int(value.q)),
        ),
    )


def central_binomial(request: NonnegativeIntegerRequest) -> IntegerResult:
    import math

    n = request.n
    return _integer_result(math.comb(2 * n, n))


def compositions(request: NonnegativePairRequest) -> IntegerResult:
    import math

    pair = request
    if pair.n == pair.k == 0:
        return _integer_result(1)
    if 0 < pair.k <= pair.n:
        return _integer_result(math.comb(pair.n - 1, pair.k - 1))
    return _integer_result(0)
