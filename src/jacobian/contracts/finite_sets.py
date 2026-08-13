"""Typed wire contracts for finite integer-set operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.exact import CanonicalInteger
from jacobian.contracts.results import ContractModel

_MAX_SET_SIZE = 128
_MAX_BINARY_SET_RESULT_SIZE = 2 * _MAX_SET_SIZE


class FiniteIntegerSet(ContractModel):
    """One finite set of canonical integers, possibly empty."""

    elements: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_SET_SIZE)

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("finite set elements must be unique")
        return self


class FiniteSetPairRequest(ContractModel):
    """Two finite integer sets supplied to a binary set operation."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class FiniteSetElementListResult(ContractModel):
    """Sorted distinct integers produced by a binary set operation."""

    elements: tuple[CanonicalInteger, ...] = Field(
        max_length=_MAX_BINARY_SET_RESULT_SIZE
    )

    @model_validator(mode="after")
    def require_sorted_unique(self) -> Self:
        values = [int(element) for element in self.elements]
        if values != sorted(values):
            raise ValueError("set element list must be sorted")
        if len(set(values)) != len(values):
            raise ValueError("set element list must be unique")
        return self


class FiniteSetCardinalityResult(ContractModel):
    """Number of distinct elements in one finite set."""

    cardinality: int = Field(ge=0, le=_MAX_BINARY_SET_RESULT_SIZE)


class FiniteSetBooleanResult(ContractModel):
    """Truth value of a finite-set predicate."""

    holds: bool
