"""Typed wire contracts for submodular optimization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational

MAX_GROUND_SET = 20


class SetFunctionEntry(ContractModel):
    """One set-function value: f(S) for a subset S of the ground set."""

    subset: tuple[int, ...] = Field(default=())
    value: CanonicalRational


class SetFunction(ContractModel):
    """A finite set function f: 2^N -> Q given as a table."""

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET)
    entries: tuple[SetFunctionEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_valid_subsets(self) -> Self:
        for entry in self.entries:
            for elem in entry.subset:
                if not (0 <= elem < self.ground_set_size):
                    raise ValueError("subset elements must be in 0..ground_set_size-1")
        return self


class SetFunctionEvalRequest(ContractModel):
    """Evaluate a set function at a given subset."""

    function: SetFunction
    subset: tuple[int, ...] = Field(default=())


class SetFunctionEvalResult(ContractModel):
    """Value of f(S) or unknown."""

    value: str
    found: bool


class MonotonicityCheckRequest(ContractModel):
    """Check if a set function is monotone non-decreasing."""

    function: SetFunction


class MonotonicityCheckResult(ContractModel):
    """Whether the function is monotone non-decreasing."""

    is_monotone: bool
    violation: str


class SubmodularityCheckRequest(ContractModel):
    """Check if a set function is submodular."""

    function: SetFunction


class SubmodularityCheckResult(ContractModel):
    """Whether the function is submodular."""

    is_submodular: bool
    violation: str


__all__ = [
    "SetFunction",
    "SetFunctionEntry",
    "SetFunctionEvalRequest",
    "SetFunctionEvalResult",
    "MonotonicityCheckRequest",
    "MonotonicityCheckResult",
    "SubmodularityCheckRequest",
    "SubmodularityCheckResult",
]
