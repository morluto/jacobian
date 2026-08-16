"""Pratt certificate construction and verification kernels.

A Pratt certificate for a prime ``p > 2`` is:
  - A primitive root ``g`` mod ``p`` (an element of multiplicative order ``p-1``).
  - The prime factorization of ``p-1``.
  - A recursive Pratt certificate for each distinct prime factor of ``p-1``.

The base case is ``p = 2``, which is prime by definition.

Verification checks:
  1. For ``p = 2``: trivially true.
  2. For ``p > 2``: ``g^(p-1) ≡ 1 (mod p)`` (Fermat's little theorem), and
     ``g^((p-1)/q) ≢ 1 (mod p)`` for every prime divisor ``q`` of ``p-1``
     (ensuring the order of ``g`` is exactly ``p-1``, i.e. ``g`` is a primitive root).
  3. Each ``q`` is certified by its own recursive Pratt certificate.
"""

from __future__ import annotations

__all__ = ["build_pratt_certificate", "verify_pratt_certificate"]

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PrattCertificate:
    """A Pratt primality certificate tree node.

    Attributes:
        prime: The prime being certified.
        witness: The primitive root mod prime (None for p=2 base case).
        cofactor_factors: Distinct prime factors of prime-1 with their exponents.
        cofactor_certificates: Pratt certificates for each distinct prime factor
            of prime-1.
    """

    prime: int
    witness: int | None = None
    cofactor_factors: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    cofactor_certificates: tuple["PrattCertificate", ...] = field(default_factory=tuple)


def build_pratt_certificate(p: int) -> PrattCertificate:
    """Build a Pratt certificate for a prime ``p``.

    Uses SymPy to find a primitive root and factor ``p-1``.
    Recursively builds certificates for the prime factors of ``p-1``.
    """
    from sympy import factorint, primitive_root

    if p < 2:
        raise ValueError(f"cannot build Pratt certificate for non-prime {p}")

    if p == 2:
        return PrattCertificate(prime=2)

    # p is an odd prime > 2
    g = int(primitive_root(p))
    factors = {int(base): int(exp) for base, exp in factorint(p - 1).items()}

    cofactor_factors: tuple[tuple[int, int], ...] = tuple(
        (base, exp) for base, exp in sorted(factors.items())
    )

    # Recursively build certificates for each prime factor of p-1.
    # Each prime factor only needs one certificate even if it appears
    # multiple times (with higher exponent).
    cofactor_certificates: list[PrattCertificate] = []
    for base, _exp in cofactor_factors:
        cofactor_certificates.append(build_pratt_certificate(base))

    return PrattCertificate(
        prime=p,
        witness=g,
        cofactor_factors=cofactor_factors,
        cofactor_certificates=tuple(cofactor_certificates),
    )


def verify_pratt_certificate(cert: PrattCertificate) -> bool:
    """Verify a Pratt certificate.

    Returns True if the certificate proves the primality of its declared prime.
    """
    if cert.prime < 2:
        return False

    if cert.prime == 2:
        return (
            cert.witness is None
            and not cert.cofactor_factors
            and not cert.cofactor_certificates
        )

    if cert.prime == 3:
        # 3-1 = 2, so the only factor is 2 (base case)
        if cert.witness is None:
            return False
        if cert.cofactor_factors != ((2, 1),):
            return False
        if len(cert.cofactor_certificates) != 1:
            return False
        if cert.cofactor_certificates[0].prime != 2:
            return False
        # Verify witness is a primitive root mod 3
        if not _verify_primitive_root(cert.witness, cert.prime, (2,)):
            return False
        return verify_pratt_certificate(cert.cofactor_certificates[0])

    # For p > 3:
    if cert.witness is None:
        return False

    # Check that the cofactor factors reconstruct p-1
    product = 1
    for base, exp in cert.cofactor_factors:
        product *= base**exp
    if product != cert.prime - 1:
        return False

    # Verify the witness is a primitive root mod prime.
    # That means: g^(p-1) ≡ 1 (mod p) and g^((p-1)/q) ≢ 1 (mod p)
    # for every prime divisor q of p-1.
    distinct_primes = tuple(base for base, _ in cert.cofactor_factors)
    if not _verify_primitive_root(cert.witness, cert.prime, distinct_primes):
        return False

    # Verify the number of cofactor certificates matches the number of
    # distinct prime factors
    if len(cert.cofactor_certificates) != len(cert.cofactor_factors):
        return False

    # Recursively verify each cofactor certificate
    for i, (base, _exp) in enumerate(cert.cofactor_factors):
        sub_cert = cert.cofactor_certificates[i]
        if sub_cert.prime != base:
            return False
        if not verify_pratt_certificate(sub_cert):
            return False

    return True


def _verify_primitive_root(g: int, p: int, distinct_prime_factors: tuple[int, ...]) -> bool:
    """Verify that ``g`` is a primitive root mod ``p``.

    ``g`` is a primitive root mod ``p`` if and only if:
    - ``g^(p-1) ≡ 1 (mod p)`` (Fermat's little theorem), and
    - ``g^((p-1)/q) ≢ 1 (mod p)`` for every prime divisor ``q`` of ``p-1``.

    The second condition ensures the multiplicative order of ``g`` is exactly ``p-1``,
    which means ``p`` has a primitive root, so ``p`` is prime.
    """
    if g <= 0 or g >= p:
        return False
    if pow(g, p - 1, p) != 1:
        return False
    for q in distinct_prime_factors:
        if pow(g, (p - 1) // q, p) == 1:
            return False
    return True
