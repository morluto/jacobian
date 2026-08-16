"""Typed wire contracts for permutation group operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalInteger

MAX_PERM_DEGREE = 64


class PermutationGroupRequest(ContractModel):
    degree: int = Field(ge=1, le=MAX_PERM_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_PERM_DEGREE
    )

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        return self


class PermutationGroupOrderResult(ContractModel):
    order: CanonicalInteger
    method: Literal["SYMPY_SCHREIER_SIMS"] = "SYMPY_SCHREIER_SIMS"


class PermutationGroupOrbitRequest(ContractModel):
    degree: int = Field(ge=1, le=MAX_PERM_DEGREE)
    generators: tuple[tuple[int, ...], ...] = Field(
        min_length=1, max_length=MAX_PERM_DEGREE
    )
    point: int = Field(ge=0, le=MAX_PERM_DEGREE - 1)

    @model_validator(mode="after")
    def require_valid_generators(self) -> Self:
        for perm in self.generators:
            if len(perm) != self.degree:
                raise ValueError("each generator must have length equal to degree")
            if sorted(perm) != list(range(self.degree)):
                raise ValueError("each generator must be a permutation of 0..n-1")
        return self


class PermutationGroupOrbitResult(ContractModel):
    orbit: tuple[int, ...] = Field(min_length=1, max_length=MAX_PERM_DEGREE)
