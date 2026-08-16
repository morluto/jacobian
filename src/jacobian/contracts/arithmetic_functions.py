"""Typed contracts for arithmetic function operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian.contracts.base import ContractModel


class ArithmeticFunctionTable(ContractModel):
    """A bounded finite table of arithmetic function values f(1), ..., f(N)."""

    values: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_table(self) -> Self:
        return self


class DirichletConvolutionRequest(ContractModel):
    """Request to compute the Dirichlet convolution (f * g)(n) for 1 <= n <= N."""

    left: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)
    right: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_same_length(self) -> Self:
        if len(self.left) != len(self.right):
            raise ValueError(
                f"left and right tables must have the same length, got "
                f"{len(self.left)} and {len(self.right)}"
            )
        return self


class DirichletConvolutionResult(ContractModel):
    """Result of a Dirichlet convolution."""

    values: tuple[StrictInt, ...]


class MobiusTransformRequest(ContractModel):
    """Request to compute the Mobius (divisor) transform of a table."""

    values: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)


class MobiusTransformResult(ContractModel):
    """Result of the Mobius transform."""

    values: tuple[StrictInt, ...]


class DirichletInverseRequest(ContractModel):
    """Request to compute the Dirichlet inverse of a table (with f(1)=1)."""

    values: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def validate_f_one(self) -> Self:
        if self.values[0] != 1:
            raise ValueError("Dirichlet inverse requires f(1) = 1")
        return self


class DirichletInverseResult(ContractModel):
    """Result of the Dirichlet inverse."""

    values: tuple[StrictInt, ...]


class SummatoryFunctionRequest(ContractModel):
    """Request to compute the summatory (prefix sum) of an arithmetic function."""

    values: tuple[StrictInt, ...] = Field(min_length=1, max_length=1024)


class SummatoryFunctionResult(ContractModel):
    """Result of the summatory function."""

    values: tuple[StrictInt, ...]
