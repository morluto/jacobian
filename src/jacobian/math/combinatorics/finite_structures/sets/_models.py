"""Typed wire contracts for finite integer-set operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, parse_canonical_integer

MAX_FINITE_SET_OPERAND_ELEMENTS = 50_000
MAX_FINITE_INTEGER_SET_ELEMENTS = 2 * MAX_FINITE_SET_OPERAND_ELEMENTS
MAX_FINITE_SET_COVERAGE_VALUES = MAX_FINITE_INTEGER_SET_ELEMENTS
_MAX_FINITE_SET_WIRE_BYTES = CanonicalLimits().max_output_bytes // 2


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"finite_set.{reason}", message)


class FiniteIntegerSet(StrictModel):
    """One finite set of canonical integers, possibly empty."""

    elements: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_INTEGER_SET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise _validation_error(
                "elements_not_unique", "finite set elements must be unique"
            )
        estimated = sum(len(value) + 3 for value in self.elements) + 64
        if estimated > _MAX_FINITE_SET_WIRE_BYTES:
            raise _validation_error(
                "wire_bytes_exceeded",
                "finite set request exceeds the "
                f"{_MAX_FINITE_SET_WIRE_BYTES}-byte transport envelope; "
                "partition the set into ≤10MiB chunks and compose",
            )
        return self


class FiniteSetPairRequest(StrictModel):
    """Two finite integer sets supplied to a binary set operation."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class FiniteSetCoverageRequest(StrictModel):
    """A finite integer scope and a bounded sequence intended to cover it once."""

    scope: FiniteIntegerSet
    values: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_SET_COVERAGE_VALUES
    )


class FiniteSetCoverageResult(StrictModel):
    """Exact diagnostics for a bounded exactly-once finite-set cover."""

    holds: bool
    missing: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_INTEGER_SET_ELEMENTS
    )
    duplicates: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_SET_COVERAGE_VALUES
    )
    outside: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_SET_COVERAGE_VALUES
    )

    @model_validator(mode="after")
    def require_canonical_diagnostics(self) -> Self:
        for name in ("missing", "duplicates", "outside"):
            values = [parse_canonical_integer(value) for value in getattr(self, name)]
            if values != sorted(set(values)):
                raise _validation_error(
                    "coverage_values_invalid",
                    f"coverage {name} values must be sorted and unique",
                )
        if self.holds != (not (self.missing or self.duplicates or self.outside)):
            raise _validation_error(
                "coverage_diagnostics_inconsistent",
                "coverage truth value and diagnostics disagree",
            )
        return self


class FiniteSetElementListResult(StrictModel):
    """Sorted distinct integers produced by a binary set operation."""

    elements: tuple[CanonicalInteger, ...] = Field(
        max_length=MAX_FINITE_INTEGER_SET_ELEMENTS
    )

    @model_validator(mode="after")
    def require_sorted_unique(self) -> Self:
        values = [parse_canonical_integer(element) for element in self.elements]
        if values != sorted(values):
            raise _validation_error(
                "elements_not_sorted", "set element list must be sorted"
            )
        if len(set(values)) != len(values):
            raise _validation_error(
                "elements_not_unique", "set element list must be unique"
            )
        return self


class FiniteSetCardinalityResult(StrictModel):
    """Number of distinct elements in one finite set."""

    cardinality: int = Field(ge=0, le=MAX_FINITE_INTEGER_SET_ELEMENTS)


class FiniteSetBooleanResult(StrictModel):
    """Truth value of a finite-set predicate."""

    holds: bool
