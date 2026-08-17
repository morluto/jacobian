"""Pure prime and arithmetic-function operation kernels."""

from __future__ import annotations

from jacobian.math.number_theory._models import (
    BooleanResult,
    IntegerValueRequest,
    IntegerValueResult,
    NonnegativeIntegerRequest,
    PositiveIntegerRequest,
    PreviousPrimeRequest,
)


def decide_prime(request: IntegerValueRequest) -> BooleanResult:
    from sympy import isprime

    return BooleanResult(holds=bool(isprime(int(request.value))))


def compute_next_prime(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    from sympy import nextprime

    return IntegerValueResult(value=str(int(nextprime(request.n))))


def compute_previous_prime(request: PreviousPrimeRequest) -> IntegerValueResult:
    from sympy import prevprime

    return IntegerValueResult(value=str(int(prevprime(request.n))))


def compute_prime_count(request: NonnegativeIntegerRequest) -> IntegerValueResult:
    from sympy import primepi

    return IntegerValueResult(value=str(int(primepi(request.n))))


def compute_nth_prime(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import prime

    return IntegerValueResult(value=str(int(prime(request.n))))


def compute_primorial(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import primorial

    return IntegerValueResult(value=str(int(primorial(request.n))))


def compute_euler_totient(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import totient

    return IntegerValueResult(value=str(int(totient(request.n))))


def compute_mobius(request: PositiveIntegerRequest) -> IntegerValueResult:
    from sympy import mobius

    return IntegerValueResult(value=str(int(mobius(request.n))))
