"""Pure divisibility and arithmetic-function operation kernels."""

from __future__ import annotations

import math

from jacobian.math.number_theory._models import (
    BooleanResult,
    DivisibilityRequest,
    ExtendedGcdResult,
    IntegerPairRequest,
    IntegerValueRequest,
    IntegerValueResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    ValuationRequest,
)


def compute_gcd(request: IntegerPairRequest) -> IntegerValueResult:
    return IntegerValueResult(
        value=str(math.gcd(int(request.left), int(request.right)))
    )


def compute_lcm(request: IntegerPairRequest) -> IntegerValueResult:
    return IntegerValueResult(
        value=str(math.lcm(int(request.left), int(request.right)))
    )


def compute_extended_gcd(request: IntegerPairRequest) -> ExtendedGcdResult:
    from sympy import gcdex

    left, right = int(request.left), int(request.right)
    x, y, divisor = gcdex(left, right)
    return ExtendedGcdResult(
        gcd=str(int(divisor)),
        left_coefficient=str(int(x)),
        right_coefficient=str(int(y)),
    )


def compute_valuation(request: ValuationRequest) -> IntegerValueResult:
    from sympy import multiplicity

    value, prime = int(request.value), int(request.prime)
    return IntegerValueResult(value=str(multiplicity(abs(prime), abs(value))))


def compute_divisor_count(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import divisor_count

    return IntegerValueResult(value=str(int(divisor_count(request.n))))


def compute_divisor_sum(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import divisor_sigma

    return IntegerValueResult(value=str(int(divisor_sigma(request.n))))


def compute_aliquot_sum(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import divisor_sigma

    return IntegerValueResult(value=str(int(divisor_sigma(request.n)) - request.n))


def decide_coprime(request: IntegerPairRequest) -> BooleanResult:
    return BooleanResult(holds=math.gcd(int(request.left), int(request.right)) == 1)


def decide_divides(request: DivisibilityRequest) -> BooleanResult:
    divisor, dividend = int(request.divisor), int(request.dividend)
    return BooleanResult(holds=dividend % divisor == 0)


def decide_even(request: IntegerValueRequest) -> BooleanResult:
    return BooleanResult(holds=int(request.value) % 2 == 0)


def decide_odd(request: IntegerValueRequest) -> BooleanResult:
    return BooleanResult(holds=int(request.value) % 2 != 0)


def decide_square(request: NonnegativeIntegerRequest) -> BooleanResult:
    return BooleanResult(holds=math.isqrt(request.n) ** 2 == request.n)


def _aliquot_relation(request: NonnegativeIntegerRequest) -> int:
    from sympy import divisor_sigma

    return int(divisor_sigma(request.n)) - request.n


def decide_perfect(request: NonnegativeIntegerRequest) -> BooleanResult:
    return BooleanResult(
        holds=bool(request.n and _aliquot_relation(request) == request.n)
    )


def decide_abundant(request: NonnegativeIntegerRequest) -> BooleanResult:
    return BooleanResult(
        holds=bool(request.n and _aliquot_relation(request) > request.n)
    )


def decide_deficient(request: NonnegativeIntegerRequest) -> BooleanResult:
    return BooleanResult(
        holds=bool(request.n and _aliquot_relation(request) < request.n)
    )
