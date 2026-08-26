"""Contracts for certified factorization and Pratt primality certificates."""

from __future__ import annotations

import math
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, StringConstraints, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer
from jacobian.math.number_theory._models import BoundedInteger, _validation_error

# SymPy factoring and recursive factoring of p - 1 are synchronous here.
MAX_CERTIFIED_FACTORIZATION_DIGITS = 30
CertifiedFactorizationInteger = Annotated[
    str,
    StringConstraints(
        pattern=r"^-?(?:0|[1-9][0-9]*)$",
        max_length=MAX_CERTIFIED_FACTORIZATION_DIGITS,
        strict=True,
    ),
]


def _verify_pratt_identities(p: int, witness: int, sub_primes: tuple[int, ...]) -> None:
    if pow(witness, p - 1, p) != 1:
        raise _validation_error(
            "pratt_witness_fails_a_p_1_1_mod_p",
            "Pratt witness fails a^(p-1) ≡ 1 (mod p)",
        )
    for q in sub_primes:
        if (p - 1) % q != 0:
            raise _validation_error(
                "sub_certificate_prime_must_divide_p_1",
                "sub-certificate prime must divide p-1",
            )
        if pow(witness, (p - 1) // q, p) == 1:
            raise _validation_error(
                "pratt_witness_fails_a_p_1_q_1_mod_p",
                "Pratt witness fails a^((p-1)/q) ≢ 1 (mod p)",
            )
    residual = p - 1
    for q in sub_primes:
        while residual % q == 0:
            residual //= q
    if residual != 1:
        raise _validation_error(
            "sub_certificates_must_exactly_cover_the_distinct_prime_factors_of_p_1",
            "sub-certificates must exactly cover the distinct prime factors of p-1",
        )


class PrattCertificateNode(StrictModel):
    """One recursively checked Pratt certificate node."""

    prime: BoundedInteger
    witness: BoundedInteger | None = None
    sub_certificates: tuple[PrattCertificateNode, ...] = Field(
        default_factory=tuple, min_length=0, max_length=256
    )

    @model_validator(mode="after")
    def require_valid_certificate(self) -> Self:
        prime = parse_canonical_integer(self.prime)
        if prime < 2:
            raise _validation_error(
                "certificate_prime_must_be_at_least_2",
                "certificate prime must be at least 2",
            )
        if prime == 2:
            if self.witness is not None:
                raise _validation_error(
                    "base_case_prime_2_has_no_witness",
                    "base case prime 2 has no witness",
                )
            if self.sub_certificates:
                raise _validation_error(
                    "base_case_prime_2_has_no_sub_certificates",
                    "base case prime 2 has no sub-certificates",
                )
            return self
        if self.witness is None:
            raise _validation_error(
                "non_base_case_certificate_requires_a_witness",
                "non-base-case certificate requires a witness",
            )
        sub_primes_str = [item.prime for item in self.sub_certificates]
        if len(set(sub_primes_str)) != len(self.sub_certificates):
            raise _validation_error(
                "sub_certificate_primes_must_be_unique",
                "sub-certificate primes must be unique",
            )
        witness = parse_canonical_integer(self.witness)
        if witness < 2 or witness >= prime:
            raise _validation_error(
                "witness_must_be_between_2_and_p_1", "witness must be between 2 and p-1"
            )
        _verify_pratt_identities(
            prime,
            witness,
            tuple(
                parse_canonical_integer(item.prime) for item in self.sub_certificates
            ),
        )
        return self


class CertifiedFactorizationRequest(StrictModel):
    """One positive integer for bounded certified factorization."""

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_composite_domain(self) -> Self:
        if parse_canonical_integer(self.value) < 2:
            raise _validation_error(
                "certified_factorization_requires_an_integer_at_least_2",
                "certified factorization requires an integer at least 2",
            )
        return self


class CertifiedFactor(StrictModel):
    """One certified prime factor with its Pratt certificate."""

    prime: BoundedInteger
    exponent: StrictInt = Field(ge=1, le=4096)
    certificate: PrattCertificateNode


class CertifiedFactorizationResult(StrictModel):
    """The complete certified prime-power factorization of one integer."""

    status: Literal["COMPLETE", "UNKNOWN"]
    value: CertifiedFactorizationInteger
    factors: tuple[CertifiedFactor, ...] = Field(min_length=0, max_length=256)
    detail: str | None = Field(default=None, max_length=1_024)

    @model_validator(mode="after")
    def bind_decomposition(self) -> Self:
        if self.status == "UNKNOWN":
            if self.factors:
                raise _validation_error(
                    "unknown_factorization_has_no_factors",
                    "an unknown factorization must not carry partial factors",
                )
            if self.detail is None:
                raise _validation_error(
                    "unknown_factorization_requires_detail",
                    "an unknown factorization must state its execution condition",
                )
            return self
        if not self.factors:
            raise _validation_error(
                "complete_factorization_requires_factors",
                "a complete factorization must carry at least one factor",
            )
        if math.prod(
            parse_canonical_integer(item.prime) ** item.exponent
            for item in self.factors
        ) != parse_canonical_integer(self.value):
            raise _validation_error(
                "factor_components_must_multiply_to_the_requested_integer",
                "factor components must multiply to the requested integer",
            )
        primes = [parse_canonical_integer(item.prime) for item in self.factors]
        if primes != sorted(primes):
            raise _validation_error(
                "factor_primes_must_be_ascending", "factor primes must be ascending"
            )
        if len(set(primes)) != len(primes):
            raise _validation_error(
                "factor_primes_must_be_unique", "factor primes must be unique"
            )
        if any(
            parse_canonical_integer(item.certificate.prime)
            != parse_canonical_integer(item.prime)
            for item in self.factors
        ):
            raise _validation_error(
                "factor_certificate_prime_must_equal_the_factor_prime",
                "factor certificate prime must equal the factor prime",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CertifiedFactorizationInteger,
        factors: tuple[CertifiedFactor, ...],
    ) -> Self:
        return cls.model_construct(
            status="COMPLETE", value=value, factors=factors, detail=None
        )

    @classmethod
    def _unknown(cls, *, value: CertifiedFactorizationInteger, detail: str) -> Self:
        return cls.model_construct(
            status="UNKNOWN", value=value, factors=(), detail=detail
        )


class PrimalityCertificateRequest(StrictModel):
    """One positive integer to be certified as prime via a Pratt certificate."""

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_candidate_domain(self) -> Self:
        if parse_canonical_integer(self.value) < 2:
            raise _validation_error(
                "primality_certificate_requires_an_integer_at_least_2",
                "primality certificate requires an integer at least 2",
            )
        return self


class PrimalityCertificateResult(StrictModel):
    """A Pratt primality certificate for one declared candidate."""

    status: Literal["CERTIFIED", "COMPOSITE"]
    value: CertifiedFactorizationInteger
    certificate: PrattCertificateNode | None = None

    @model_validator(mode="after")
    def bind_result(self) -> Self:
        certificate = self.certificate
        if self.status == "CERTIFIED":
            if certificate is None:
                raise _validation_error(
                    "certified_status_requires_a_certificate",
                    "CERTIFIED status requires a certificate",
                )
            if parse_canonical_integer(certificate.prime) != parse_canonical_integer(
                self.value
            ):
                raise _validation_error(
                    "certificate_prime_must_match_the_candidate_value",
                    "certificate prime must match the candidate value",
                )
        elif certificate is not None:
            raise _validation_error(
                "composite_status_must_not_carry_a_certificate",
                "COMPOSITE status must not carry a certificate",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: Literal["CERTIFIED", "COMPOSITE"],
        value: CertifiedFactorizationInteger,
        certificate: PrattCertificateNode | None = None,
    ) -> Self:
        return cls.model_construct(status=status, value=value, certificate=certificate)


PrattCertificateNode.model_rebuild()

__all__ = [
    "CertifiedFactor",
    "CertifiedFactorizationInteger",
    "CertifiedFactorizationRequest",
    "CertifiedFactorizationResult",
    "PrattCertificateNode",
    "PrimalityCertificateRequest",
    "PrimalityCertificateResult",
]
