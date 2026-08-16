"""Typed wire contracts for recurrence solving."""
from __future__ import annotations
from typing import Literal
from pydantic import Field
from jacobian.contracts.base import ContractModel


class RecurrenceFindRequest(ContractModel):
    """Find the minimal linear recurrence of a sequence over QQ."""
    sequence: tuple[str, ...] = Field(min_length=2, max_length=256)


class RecurrenceFindResult(ContractModel):
    """The minimal linear recurrence coefficients."""
    coefficients: tuple[str, ...]
    order: int = Field(ge=1, le=128)
    method: Literal["RATIONAL_INTERPOLATION"] = "RATIONAL_INTERPOLATION"


class ClosedFormRequest(ContractModel):
    """Compute the closed-form solution of a linear recurrence."""
    characteristic_coefficients: tuple[str, ...] = Field(min_length=1, max_length=64)
    initial_values: tuple[str, ...] = Field(min_length=1, max_length=64)


class ClosedFormResult(ContractModel):
    """The closed-form solution as a SymPy expression string."""
    expression: str
    method: Literal["SYMPY_RSOLVE"] = "SYMPY_RSOLVE"
