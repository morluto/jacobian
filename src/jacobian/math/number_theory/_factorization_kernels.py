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


def _replayed_divisors(value: int, *, proper: bool) -> tuple[str, ...]:
    from sympy import divisors

    return tuple(str(item) for item in divisors(abs(value), proper=proper))


def enumerate_divisors(request: FactorizationRequest) -> DivisorListResult:
    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(
        value=request.value,
        divisors=_replayed_divisors(value, proper=False),
        convention="ALL_POSITIVE_DIVISORS",
    )


def enumerate_proper_divisors(request: FactorizationRequest) -> DivisorListResult:
    value = int(request.value)
    if value == 0:
        raise ValueError("zero has infinitely many divisors")
    return DivisorListResult(
        value=request.value,
        divisors=_replayed_divisors(value, proper=True),
        convention="PROPER_DIVISORS",
    )


def factorize_primes(request: FactorizationRequest) -> PrimeFactorizationResult:
    from sympy import factorint

    value = int(request.value)
    if value == 0:
        raise ValueError("zero has no finite prime factorization")
    return PrimeFactorizationResult(
        value=request.value,
        factors=tuple(
            PrimePower(prime=str(prime), power=int(power))
            for prime, power in sorted(factorint(abs(value)).items())
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
