"""Typed wire contracts for finite integer-set operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

_MAX_SET_SIZE = 128
_MAX_BINARY_SET_RESULT_SIZE = 2 * _MAX_SET_SIZE
_MAX_COVERAGE_VALUES = 2 * _MAX_SET_SIZE


class FiniteIntegerSet(StrictModel):
    """One finite set of canonical integers, possibly empty."""

    elements: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_SET_SIZE)

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("finite set elements must be unique")
        return self


class FiniteSetPairRequest(StrictModel):
    """Two finite integer sets supplied to a binary set operation."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class FiniteSetCoverageRequest(StrictModel):
    """A finite integer scope and a bounded sequence intended to cover it once."""

    scope: FiniteIntegerSet
    values: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_COVERAGE_VALUES)


class FiniteSetCoverageResult(StrictModel):
    """Exact diagnostics for a bounded exactly-once finite-set cover."""

    holds: bool
    missing: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_SET_SIZE)
    duplicates: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_COVERAGE_VALUES)
    outside: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_COVERAGE_VALUES)

    @model_validator(mode="after")
    def require_canonical_diagnostics(self) -> Self:
        for name in ("missing", "duplicates", "outside"):
            values = [parse_canonical_integer(value) for value in getattr(self, name)]
            if values != sorted(set(values)):
                raise ValueError(f"coverage {name} values must be sorted and unique")
        if self.holds != (not (self.missing or self.duplicates or self.outside)):
            raise ValueError("coverage truth value and diagnostics disagree")
        return self


class FiniteSetElementListResult(StrictModel):
    """Sorted distinct integers produced by a binary set operation."""

    elements: tuple[CanonicalInteger, ...] = Field(
        max_length=_MAX_BINARY_SET_RESULT_SIZE
    )

    @model_validator(mode="after")
    def require_sorted_unique(self) -> Self:
        values = [parse_canonical_integer(element) for element in self.elements]
        if values != sorted(values):
            raise ValueError("set element list must be sorted")
        if len(set(values)) != len(values):
            raise ValueError("set element list must be unique")
        return self


class FiniteSetCardinalityResult(StrictModel):
    """Number of distinct elements in one finite set."""

    cardinality: int = Field(ge=0, le=_MAX_BINARY_SET_RESULT_SIZE)


class FiniteSetBooleanResult(StrictModel):
    """Truth value of a finite-set predicate."""

    holds: bool
