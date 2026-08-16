"""Typed wire contracts for certified integer factoring and primality certificates."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger

MAX_CERTIFIED_FACTORS = 256
MAX_PRATT_CERTIFICATE_DEPTH = 512
MAX_CERTIFIED_INTEGER_DIGITS = 4_300


class CertifiedFactorRequest(ContractModel):
    """One bounded positive integer for certified factorization."""

    n: CanonicalInteger = Field(min_length=1, max_length=MAX_CERTIFIED_INTEGER_DIGITS)

    @model_validator(mode="after")
    def require_positive_nonzero(self) -> Self:
        if parse_canonical_integer(self.n) < 2:
            raise ValueError("n must be an integer greater than 1")
        return self


class PrimalityCertificateRequest(ContractModel):
    """One bounded integer for which a Pratt certificate is requested."""

    p: CanonicalInteger = Field(min_length=1, max_length=MAX_CERTIFIED_INTEGER_DIGITS)

    @model_validator(mode="after")
    def require_prime_candidate(self) -> Self:
        value = parse_canonical_integer(self.p)
        if value < 2:
            raise ValueError("p must be at least 2 for a primality certificate")
        return self


class PrattCertificateNode(ContractModel):
    """One node of a recursive Pratt (primality) certificate tree.

    A Pratt certificate for a prime ``p`` consists of:
    - A primitive root ``witness`` mod ``p`` (an element of multiplicative order ``p-1``).
    - The factorization of ``p-1`` into prime powers.
    - A recursive Pratt certificate for each distinct prime factor of ``p-1``.

    The base case is ``p = 2``, which is prime by definition and has an empty
    factor list and no recursive certificates.
    """

    prime: CanonicalInteger
    witness: CanonicalInteger | None = Field(
        default=None,
        description="Primitive root mod p; None for the base case p=2.",
    )
    cofactor_factors: tuple["PrattFactor", ...] = Field(
        default_factory=tuple,
        max_length=MAX_PRATT_CERTIFICATE_DEPTH,
    )
    cofactor_certificates: tuple["PrattCertificateNode", ...] = Field(
        default_factory=tuple,
        max_length=MAX_PRATT_CERTIFICATE_DEPTH,
    )

    @model_validator(mode="after")
    def require_base_case_consistency(self) -> Self:
        if parse_canonical_integer(self.prime) == 2:
            if self.witness is not None:
                raise ValueError("Pratt certificate for p=2 must not have a witness")
            if self.cofactor_factors or self.cofactor_certificates:
                raise ValueError(
                    "Pratt certificate for p=2 must have empty cofactor fields"
                )
        else:
            if self.witness is None:
                raise ValueError(
                    "Pratt certificate for p>2 must have a primitive root witness"
                )
        return self


class PrattFactor(ContractModel):
    """One prime-power factor of p-1 in a Pratt certificate."""

    prime: CanonicalInteger
    exponent: int = Field(ge=1)


# Re-create PrattCertificateNode's forward references now that PrattFactor exists
PrattCertificateNode.model_rebuild()


class CertifiedFactorResult(ContractModel):
    """The complete certified factorization of one positive integer.

    Each factor is a prime with its multiplicity, plus a Pratt certificate
    proving the factor's primality.  The product of ``prime^exponent`` over
    all factors must equal the original integer ``n``.
    """

    factors: tuple["CertifiedFactor", ...] = Field(
        min_length=1, max_length=MAX_CERTIFIED_FACTORS
    )
    method: Literal["SYMPY_FACTORINT_WITH_PRATT"] = "SYMPY_FACTORINT_WITH_PRATT"

    @model_validator(mode="after")
    def require_unique_primes(self) -> Self:
        primes = [factor.prime for factor in self.factors]
        if len(set(primes)) != len(primes):
            raise ValueError("certified factor primes must be unique")
        return self


class CertifiedFactor(ContractModel):
    """One prime factor with its exponent and Pratt primality certificate."""

    prime: CanonicalInteger
    exponent: int = Field(ge=1)
    certificate: PrattCertificateNode


class PrimalityCertificateResult(ContractModel):
    """A Pratt primality certificate for one declared prime."""

    prime: CanonicalInteger
    certificate: PrattCertificateNode
    status: Literal["PRIME"] = "PRIME"
    method: Literal["PRATT_CERTIFICATE"] = "PRATT_CERTIFICATE"

    @model_validator(mode="after")
    def require_prime_matches_certificate(self) -> Self:
        if self.prime != self.certificate.prime:
            raise ValueError(
                "primality certificate prime must match the declared prime"
            )
        return self
