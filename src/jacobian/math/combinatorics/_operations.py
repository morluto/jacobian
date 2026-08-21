"""Exact combinatorics operations backed by maintained SymPy and stdlib APIs."""

from __future__ import annotations

from functools import reduce
from operator import mul

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.combinatorics import operations as native
from jacobian.math.combinatorics._models import (
    BinomialRequest,
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


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=format_canonical_integer(value))


def factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    import math

    n = request.n
    return _integer_result(math.factorial(n))


def double_factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.double_factorial(request.n))


def derangements(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.derangement_number(request.n))


def binomial(request: BinomialRequest) -> IntegerResult:
    import math

    pair = request
    if pair.k > pair.n:
        return _integer_result(0)
    return _integer_result(math.comb(pair.n, pair.k))


def multinomial(request: IntegerListRequest) -> IntegerResult:
    import math

    values = [parse_canonical_integer(value) for value in request.values]
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
    pair = request
    return _integer_result(native.stirling_first(pair.n, pair.k))


def stirling_second(request: NonnegativePairRequest) -> IntegerResult:
    pair = request
    return _integer_result(native.stirling_second(pair.n, pair.k))


def bell(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.bell_number(request.n))


def catalan(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.catalan_number(request.n))


def partition_number(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.partition_number(request.n))


def enumerate_integer_partitions(
    request: IntegerPartitionEnumerationRequest,
) -> IntegerPartitionEnumerationResult:
    """Enumerate all bounded partitions using ``sympy.utilities.partitions``."""
    value = request
    return IntegerPartitionEnumerationResult(
        n=value.n,
        max_parts=value.max_parts,
        partitions=native.integer_partitions(value.n, max_parts=value.max_parts),
    )


def fibonacci(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.fibonacci_number(request.n))


def fibonacci_pair(request: FibonacciPairRequest) -> FibonacciPairResult:
    """Compute two consecutive Fibonacci values."""
    n = request.n
    return FibonacciPairResult(
        n=n,
        f_n=str(native.fibonacci_number(n)),
        f_n_plus_one=str(native.fibonacci_number(n + 1)),
    )


def lucas(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.lucas_number(request.n))


def motzkin(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.motzkin_number(request.n))


def bernoulli(request: NonnegativeIntegerRequest) -> RationalResult:
    value = native.bernoulli_number(request.n)
    return RationalResult(
        value=CanonicalRational(
            num=format_canonical_integer(value.numerator),
            den=format_canonical_integer(value.denominator),
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
