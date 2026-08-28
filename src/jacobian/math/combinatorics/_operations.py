"""Exact combinatorics operations backed by maintained SymPy and stdlib APIs."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.combinatorics import operations as native
from jacobian.math.combinatorics._counting_models import (
    BinomialRequest,
    IntegerListRequest,
)
from jacobian.math.combinatorics._models import (
    IntegerResult,
    NonnegativeIntegerRequest,
    NonnegativePairRequest,
    RationalResult,
)
from jacobian.math.combinatorics._partition_models import (
    IntegerPartitionEnumerationRequest,
    IntegerPartitionEnumerationResult,
)
from jacobian.math.combinatorics._recurrence_models import (
    FibonacciPairRequest,
    FibonacciPairResult,
)


def _integer_result(value: int) -> IntegerResult:
    return IntegerResult(value=format_canonical_integer(value))


def factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.factorial(request.n))


def double_factorial(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.double_factorial(request.n))


def derangements(request: NonnegativeIntegerRequest) -> IntegerResult:
    return _integer_result(native.derangement_number(request.n))


def binomial(request: BinomialRequest) -> IntegerResult:
    return _integer_result(native.binomial(request.n, request.k))


def multinomial(request: IntegerListRequest) -> IntegerResult:
    values = tuple(parse_canonical_integer(value) for value in request.values)
    return _integer_result(native.multinomial(values))


def permutations(request: NonnegativePairRequest) -> IntegerResult:
    return _integer_result(native.permutations(request.n, request.k))


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
    return _integer_result(native.central_binomial(request.n))


def compositions(request: NonnegativePairRequest) -> IntegerResult:
    return _integer_result(native.compositions(request.n, request.k))
