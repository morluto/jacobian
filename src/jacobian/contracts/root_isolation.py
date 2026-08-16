"""Typed wire contracts for polynomial root isolation and algebraic number comparison."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class UnivariatePolynomialRequest(ContractModel):
    coefficients_descending: tuple[CanonicalRational, ...] = Field(
        min_length=2, max_length=64
    )

    @model_validator(mode="after")
    def require_nonzero_leading(self) -> Self:
        if self.coefficients_descending[0] == CanonicalRational(num="0", den="1"):
            raise ValueError("leading coefficient must be nonzero")
        return self


class RootIsolationResult(ContractModel):
    """Real roots with rational isolating intervals."""

    roots: tuple[tuple[CanonicalRational, CanonicalRational], ...]
    multiplicities: tuple[int, ...]
    convention: Literal["SYMPY_REAL_ROOTS"] = "SYMPY_REAL_ROOTS"


class AlgebraicNumberInput(ContractModel):
    polynomial: tuple[CanonicalRational, ...] = Field(min_length=2, max_length=64)
    isolating_interval_lower: CanonicalRational
    isolating_interval_upper: CanonicalRational


class AlgebraicCompareRequest(ContractModel):
    left: AlgebraicNumberInput
    right: AlgebraicNumberInput


class AlgebraicCompareResult(ContractModel):
    order: Literal["LT", "EQ", "GT"]
