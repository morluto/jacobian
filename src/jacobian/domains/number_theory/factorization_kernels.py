"""Worker-safe kernels for bounded factorization-derived operations."""

from __future__ import annotations

import math

from jacobian.contracts.number_theory import (
    ArithmeticFunctionRequest,
    BooleanResult,
    DivisorListResult,
    FactorizationRequest,
    IntegerValueResult,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimeFactorizationResult,
    PrimePower,
)


def enumerate_divisors(request: FactorizationRequest) -> DivisorListResult:
    from sympy import divisors

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(divisors=tuple(str(item) for item in divisors(abs(value))))


def enumerate_proper_divisors(request: FactorizationRequest) -> DivisorListResult:
    from sympy import divisors

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(
        divisors=tuple(str(item) for item in divisors(abs(value), proper=True))
    )


def factorize_primes(request: FactorizationRequest) -> PrimeFactorizationResult:
    from sympy import factorint

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has no finite prime factorization")
    return PrimeFactorizationResult(
        factors=tuple(
            PrimePower(prime=str(prime), power=int(power))
            for prime, power in sorted(factorint(abs(value)).items())
        )
    )


def decide_powerful(request: PowerfulNumberRequest) -> PowerfulNumberResult:
    from sympy import factorint

    factors = sorted(factorint(int(request.value)).items())
    return PowerfulNumberResult(
        semantics_version="powerful-number.prime-exponents-at-least-two.v1",
        is_powerful=not any(power < 2 for _, power in factors),
        factors=tuple(
            PrimePower(prime=str(prime), power=int(power)) for prime, power in factors
        ),
        violating_primes=tuple(
            str(prime) for prime, power in factors if int(power) < 2
        ),
    )


def decide_squarefree(request: ArithmeticFunctionRequest) -> BooleanResult:
    from sympy import factorint

    if request.n == 0:
        return BooleanResult(holds=False)
    return BooleanResult(
        holds=all(power == 1 for power in factorint(request.n).values())
    )


def compute_radical(request: ArithmeticFunctionRequest) -> IntegerValueResult:
    from sympy import factorint

    return IntegerValueResult(value=str(math.prod(factorint(request.n))))
