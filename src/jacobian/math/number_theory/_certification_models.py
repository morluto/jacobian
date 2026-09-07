"""Contracts for certified factorization and Pratt primality certificates."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.number_theory._models import _validation_error

# SymPy factoring and recursive factoring of p - 1 are synchronous here.
MAX_CERTIFIED_FACTORIZATION_DIGITS = 30
CertifiedFactorizationInteger = Annotated[
    int, DecimalIntegerEncoding(max_digits=MAX_CERTIFIED_FACTORIZATION_DIGITS)
]


class PrattCertificateNode(StrictModel):
    """One Pratt witness claim; parsing checks its shape, not primality."""

    prime: CertifiedFactorizationInteger
    witness: CertifiedFactorizationInteger | None = None
    factors: tuple[PrattCertificateFactor, ...] = Field(
        default_factory=tuple, min_length=0, max_length=256
    )


class PrattCertificateFactor(StrictModel):
    """One claimed prime-power factor of a Pratt node's ``prime - 1``."""

    prime: CertifiedFactorizationInteger
    exponent: StrictInt = Field(ge=1, le=4096)
    certificate: PrattCertificateNode


class CertifiedFactorizationRequest(StrictModel):
    """One positive integer for bounded certified factorization."""

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_composite_domain(self) -> Self:
        if self.value < 2:
            raise _validation_error(
                "certified_factorization_requires_an_integer_at_least_2",
                "certified factorization requires an integer at least 2",
            )
        return self


class CertifiedFactor(StrictModel):
    """One certified prime factor with its Pratt certificate."""

    prime: CertifiedFactorizationInteger
    exponent: StrictInt = Field(ge=1, le=4096)
    certificate: PrattCertificateNode


class CertifiedFactorizationResult(StrictModel):
    """The complete certified prime-power factorization of one integer."""

    status: Literal["COMPLETE"] = "COMPLETE"
    value: CertifiedFactorizationInteger
    factors: tuple[CertifiedFactor, ...] = Field(min_length=1, max_length=256)

    @classmethod
    def _from_kernel(
        cls,
        *,
        value: CertifiedFactorizationInteger,
        factors: tuple[CertifiedFactor, ...],
    ) -> CertifiedFactorizationResult:
        return cls.model_construct(status="COMPLETE", value=value, factors=factors)


class PrimalityCertificateRequest(StrictModel):
    """One positive integer to be certified as prime via a Pratt certificate."""

    value: CertifiedFactorizationInteger

    @model_validator(mode="after")
    def require_candidate_domain(self) -> Self:
        if self.value < 2:
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

    @classmethod
    def _from_kernel(
        cls,
        *,
        status: Literal["CERTIFIED", "COMPOSITE"],
        value: CertifiedFactorizationInteger,
        certificate: PrattCertificateNode | None = None,
    ) -> PrimalityCertificateResult:
        return cls.model_construct(status=status, value=value, certificate=certificate)


PrattCertificateNode.model_rebuild()
PrattCertificateFactor.model_rebuild()

__all__ = [
    "CertifiedFactor",
    "CertifiedFactorizationInteger",
    "CertifiedFactorizationRequest",
    "CertifiedFactorizationResult",
    "PrattCertificateFactor",
    "PrattCertificateNode",
    "PrimalityCertificateRequest",
    "PrimalityCertificateResult",
]
