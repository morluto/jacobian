"""Worker-safe kernels for bounded factorization-derived operations."""

from __future__ import annotations

import math

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.number_theory._models import (
    ArithmeticFunctionRequest,
    BooleanResult,
    CertifiedFactor,
    CertifiedFactorizationRequest,
    CertifiedFactorizationResult,
    DivisorListResult,
    FactorizationRequest,
    IntegerValueResult,
    PowerfulNumberRequest,
    PowerfulNumberResult,
    PrattCertificateNode,
    PrimalityCertificateRequest,
    PrimalityCertificateResult,
    PrimeFactorizationResult,
    PrimePower,
)

# ---------------------------------------------------------------------------
# Pratt certificate construction and verification
# ---------------------------------------------------------------------------

_PRATT_BASE_PRIME = 2


def _build_pratt_certificate(prime: int) -> PrattCertificateNode:
    """Construct a Pratt certificate for one known prime.

    The base case is ``prime == 2``.  For ``prime > 2`` we search for a witness
    ``a`` such that ``a^(prime-1) ≡ 1 (mod prime)`` and ``a^((prime-1)/q) ≢ 1
    (mod prime)`` for every prime factor ``q`` of ``prime - 1``.  Each such
    ``q`` is then recursively certified.
    """
    if prime == _PRATT_BASE_PRIME:
        return PrattCertificateNode(prime="2")

    from sympy import factorint, primitive_root

    prime_minus_one = prime - 1
    factors_of_pmo = sorted(factorint(prime_minus_one).items())

    # A primitive root modulo ``prime`` is guaranteed to satisfy the Pratt
    # witness condition: its multiplicative order is exactly ``prime - 1``,
    # so ``a^((prime-1)/q) ≢ 1 (mod prime)`` for every prime ``q | prime-1``.
    witness = int(primitive_root(prime))

    sub_certificates = tuple(
        _build_pratt_certificate(int(q)) for q, _ in factors_of_pmo
    )
    return PrattCertificateNode(
        prime=format_canonical_integer(prime),
        witness=format_canonical_integer(witness),
        sub_certificates=sub_certificates,
    )


def compute_pratt_certificate(
    request: PrimalityCertificateRequest,
) -> PrimalityCertificateResult:
    """Produce a Pratt primality certificate for one declared candidate.

    Returns ``COMPOSITE`` (no certificate) when the candidate is not prime.
    """
    from sympy import isprime

    value = parse_canonical_integer(request.value)
    if not isprime(value):
        return PrimalityCertificateResult(status="COMPOSITE", value=request.value)
    return PrimalityCertificateResult(
        status="CERTIFIED",
        value=request.value,
        certificate=_build_pratt_certificate(value),
    )


# ---------------------------------------------------------------------------
# Subexponential certified factorization
# ---------------------------------------------------------------------------


def factorize_certified(
    request: CertifiedFactorizationRequest,
) -> CertifiedFactorizationResult:
    """Factor one bounded integer using subexponential methods.

    Backed by ``sympy.ntheory.factorint`` without a trial-division limit, so
    Pollard rho, Pollard p-1, and ECM are all available.  Each prime factor
    carries an independent Pratt primality certificate.
    """
    from sympy import factorint

    value = parse_canonical_integer(request.value)
    decomposition = sorted(factorint(value).items())
    factors = tuple(
        CertifiedFactor(
            prime=format_canonical_integer(int(prime)),
            exponent=int(exponent),
            certificate=_build_pratt_certificate(int(prime)),
        )
        for prime, exponent in decomposition
    )
    return CertifiedFactorizationResult(
        status="COMPLETE",
        value=request.value,
        factors=factors,
    )


# ---------------------------------------------------------------------------
# Divisor and factorization-derived operations
# ---------------------------------------------------------------------------


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
