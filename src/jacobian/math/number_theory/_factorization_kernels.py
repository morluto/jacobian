"""Worker-safe kernels for bounded factorization-derived operations."""

from __future__ import annotations

import math

from jacobian.math.number_theory._models import (
    ArithmeticFunctionRequest,
    BooleanResult,
    BudgetedFactorizationRequest,
    BudgetedFactorizationResult,
    CertifiedFactorComponent,
    DivisorListResult,
    FactorizationRequest,
    IntegerValueResult,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrimeFactorizationResult,
    PrimePower,
)


def factorize_with_budget(
    request: BudgetedFactorizationRequest,
) -> BudgetedFactorizationResult:
    """Factor a small integer and classify each bounded component exactly."""
    from flint import fmpz
    from sympy import factorint

    from jacobian.canonical import format_canonical_integer, parse_canonical_integer

    value = parse_canonical_integer(request.value)
    decomposition = sorted(factorint(value, limit=request.factor_limit).items())
    factors = tuple(
        CertifiedFactorComponent(
            value=format_canonical_integer(int(factor)),
            exponent=int(exponent),
            status=(
                "CERTIFIED_PRIME"
                if fmpz(int(factor)).is_prime()
                else "UNFACTORED_COMPOSITE"
            ),
        )
        for factor, exponent in decomposition
    )
    return BudgetedFactorizationResult(
        status="COMPLETE"
        if all(item.status == "CERTIFIED_PRIME" for item in factors)
        else "INCOMPLETE",
        value=request.value,
        factor_limit=request.factor_limit,
        factors=factors,
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
