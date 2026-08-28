"""Pure divisibility and arithmetic-function operation kernels."""

from __future__ import annotations

import math

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.number_theory._divisibility_models import (
    DivisibilityRequest,
    ExtendedGcdResult,
    IntegerPairRequest,
    ValuationRequest,
)
from jacobian.math.number_theory._integer_models import (
    BooleanResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory.arithmetic.values import IntegerValue


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"number_theory.{code}", message=message
    )


def _admit_divisor(request: DivisibilityRequest) -> None:
    if int(request.divisor) == 0:
        _domain_error(
            ("divisor",), "divisor_must_be_nonzero", "divisor must be nonzero"
        )


def _admit_valuation(request: ValuationRequest) -> None:
    if int(request.value) == 0:
        _domain_error(
            ("value",),
            "valuation_requires_nonzero_value",
            "valuation requires nonzero value",
        )
    from sympy import isprime

    if int(request.prime) < 2 or not isprime(int(request.prime)):
        _domain_error(
            ("prime",),
            "valuation_requires_a_prime_absolute_base_2",
            "valuation requires a prime absolute base >= 2",
        )


def compute_gcd(request: IntegerPairRequest) -> IntegerValue:
    return IntegerValue(value=str(math.gcd(int(request.left), int(request.right))))


def compute_lcm(request: IntegerPairRequest) -> IntegerValue:
    return IntegerValue(value=str(math.lcm(int(request.left), int(request.right))))


def compute_extended_gcd(request: IntegerPairRequest) -> ExtendedGcdResult:
    from sympy import gcdex

    left, right = int(request.left), int(request.right)
    x, y, divisor = gcdex(left, right)
    return ExtendedGcdResult(
        gcd=str(int(divisor)),
        left_coefficient=str(int(x)),
        right_coefficient=str(int(y)),
    )


def compute_valuation(request: ValuationRequest) -> IntegerValue:
    from sympy import multiplicity

    _admit_valuation(request)
    value, prime = int(request.value), int(request.prime)
    return IntegerValue(value=str(multiplicity(abs(prime), abs(value))))


def compute_divisor_count(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import divisor_count

    return IntegerValue(value=str(int(divisor_count(request.n))))


def compute_divisor_sum(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import divisor_sigma

    return IntegerValue(value=str(int(divisor_sigma(request.n))))


def compute_aliquot_sum(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import divisor_sigma

    return IntegerValue(value=str(int(divisor_sigma(request.n)) - request.n))


def decide_coprime(request: IntegerPairRequest) -> BooleanResult:
    return BooleanResult(holds=math.gcd(int(request.left), int(request.right)) == 1)


def decide_divides(request: DivisibilityRequest) -> BooleanResult:
    _admit_divisor(request)
    divisor, dividend = int(request.divisor), int(request.dividend)
    return BooleanResult(holds=dividend % divisor == 0)


def decide_even(request: IntegerValue) -> BooleanResult:
    return BooleanResult(holds=int(request.value[-1]) % 2 == 0)


def decide_odd(request: IntegerValue) -> BooleanResult:
    return BooleanResult(holds=int(request.value[-1]) % 2 != 0)


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
