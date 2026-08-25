"""Pure prime and arithmetic-function operation kernels."""

from __future__ import annotations

from jacobian.math.arithmetic.values import IntegerValue
from jacobian.math.number_theory._models import (
    BooleanResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
)
from jacobian.math.number_theory._prime_models import (
    PreviousPrimeRequest,
    PrimalityRequest,
    PrimorialRequest,
    PrimorialResult,
)


def decide_prime(request: PrimalityRequest) -> BooleanResult:
    from sympy import isprime

    return BooleanResult(holds=bool(isprime(int(request.value))))


def compute_next_prime(request: NonnegativeIntegerRequest) -> IntegerValue:
    from sympy import nextprime

    return IntegerValue(value=str(int(nextprime(request.n))))


def compute_previous_prime(request: PreviousPrimeRequest) -> IntegerValue:
    from sympy import prevprime

    return IntegerValue(value=str(int(prevprime(request.n))))


def compute_prime_count(request: NonnegativeIntegerRequest) -> IntegerValue:
    from sympy import primepi

    return IntegerValue(value=str(int(primepi(request.n))))


def compute_nth_prime(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import prime

    return IntegerValue(value=str(int(prime(request.n))))


def compute_primorial(request: PrimorialRequest) -> PrimorialResult:
    from sympy import primorial

    return PrimorialResult(value=str(int(primorial(request.n))))


def compute_euler_totient(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import totient

    return IntegerValue(value=str(int(totient(request.n))))


def compute_mobius(request: PositiveIntegerRequest) -> IntegerValue:
    from sympy import mobius

    return IntegerValue(value=str(int(mobius(request.n))))
