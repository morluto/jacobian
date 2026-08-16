"""Typed wire contracts for certified integer factoring."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger

MAX_FACTORS = 64


class CertifiedFactorRequest(ContractModel):
    n: CanonicalInteger = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def require_positive(self) -> Self:
        from jacobian.canonical import parse_canonical_integer

        if parse_canonical_integer(self.n) <= 0:
            raise ValueError("n must be a positive integer")
        return self


class CertifiedFactorResult(ContractModel):
    factors: tuple[tuple[CanonicalInteger, int], ...] = Field(
        min_length=1, max_length=MAX_FACTORS
    )
    method: Literal["SYMPY_FACTORINT"] = "SYMPY_FACTORINT"
