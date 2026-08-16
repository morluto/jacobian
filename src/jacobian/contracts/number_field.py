"""Typed wire contracts for number field operations."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from jacobian.contracts.base import ContractModel


class NumberFieldRequest(ContractModel):
    """A number field Q(alpha) defined by a minimal polynomial."""

    coefficients_descending: tuple[str, ...] = Field(min_length=2, max_length=32)
    variable: str = Field(min_length=1, max_length=10)


class NumberFieldDiscriminantResult(ContractModel):
    discriminant: str
    method: Literal["SYMPY_NUMBER_FIELD"] = "SYMPY_NUMBER_FIELD"


class NumberFieldRingOfIntegersResult(ContractModel):
    integral_basis: tuple[str, ...]
    method: Literal["SYMPY_NUMBER_FIELD"] = "SYMPY_NUMBER_FIELD"
