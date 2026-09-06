"""Typed wire contracts for submodular optimization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

# The checks run the local characterizations: monotonicity scans n*2^n
# covering relations and submodularity C(n,2)*2^n local pairs, both with
# early exit.  A ground set of size 16 keeps the complete table at 65,536
# entries and the worst-case full scan at about 8M exact pair checks.
#
# Scan work also scales with coefficient height: every inequality subtracts
# and compares exact Fractions.  The two check requests therefore bound each
# value to 128 numerator/denominator digits, keeping every one of the ~8M
# comparisons on small big-ints so the documented seconds-scale envelope
# holds.  This is scan-specific admission: the shared entry type stays at
# the canonical ceiling so the single-lookup evaluator can return any exact
# representable value.
MAX_GROUND_SET = 16
MAX_SUBMODULAR_SCAN_VALUE_DIGITS = 128


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by submodular-opt contracts."""

    return PydanticCustomError(f"submodular_opt.{reason}", message)


class SetFunctionEntry(StrictModel):
    """One set-function value: f(S) for a subset S of the ground set."""

    subset: tuple[int, ...] = Field(default=())
    value: CanonicalRational


class SetFunction(StrictModel):
    """A finite set function f: 2^N -> Q given as a table."""

    ground_set_size: int = Field(ge=0, le=MAX_GROUND_SET)
    entries: tuple[SetFunctionEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_complete_table(self) -> Self:
        expected = 1 << self.ground_set_size
        if len(self.entries) != expected:
            raise _validation_error(
                "table_entry_count_mismatch",
                "set function table must contain exactly one value per subset",
            )
        seen: set[tuple[int, ...]] = set()
        for entry in self.entries:
            if len(entry.subset) != len(set(entry.subset)):
                raise _validation_error(
                    "subset_elements_not_unique", "subset elements must be unique"
                )
            for elem in entry.subset:
                if not (0 <= elem < self.ground_set_size):
                    raise _validation_error(
                        "subset_element_out_of_range",
                        "subset elements must be in 0..ground_set_size-1",
                    )
            key = tuple(sorted(entry.subset))
            if key in seen:
                raise _validation_error(
                    "table_subsets_not_unique", "set function subsets must be unique"
                )
            seen.add(key)
        if len(seen) != expected:
            raise _validation_error(
                "table_missing_subset",
                "set function table must contain every subset of the ground set",
            )
        return self


class SetFunctionEvalRequest(StrictModel):
    """Evaluate a set function at a given subset."""

    function: SetFunction
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if len(self.subset) != len(set(self.subset)):
            raise _validation_error(
                "subset_elements_not_unique", "subset elements must be unique"
            )
        for elem in self.subset:
            if not (0 <= elem < self.function.ground_set_size):
                raise _validation_error(
                    "subset_element_out_of_range",
                    "subset elements must be in 0..ground_set_size-1",
                )
        return self


class SetFunctionEvalResult(StrictModel):
    """Exact canonical value of ``f(S)`` bound to its source and query."""

    function: SetFunction
    subset: tuple[int, ...] = Field(default=())
    value: CanonicalRational


class MonotonicityViolation(StrictModel):
    """One source-bound covering inequality that fails monotonicity."""

    subset: tuple[int, ...] = Field(default=())
    added_element: int = Field(ge=0, le=MAX_GROUND_SET - 1)
    lower_value: CanonicalRational
    upper_value: CanonicalRational


class MonotonicityCheckRequest(StrictModel):
    """Check if a set function is monotone non-decreasing."""

    function: SetFunction


class SubmodularityViolation(StrictModel):
    """One source-bound local submodular inequality that fails."""

    subset: tuple[int, ...] = Field(default=())
    first_element: int = Field(ge=0, le=MAX_GROUND_SET - 1)
    second_element: int = Field(ge=0, le=MAX_GROUND_SET - 1)
    left_sum: CanonicalRational
    right_sum: CanonicalRational


class MonotonicityCheckResult(StrictModel):
    """Whether the function is monotone non-decreasing."""

    function: SetFunction
    is_monotone: bool
    violation: MonotonicityViolation | None = None


class SubmodularityCheckRequest(StrictModel):
    """Check if a set function is submodular."""

    function: SetFunction


class SubmodularityCheckResult(StrictModel):
    """Whether the function is submodular."""

    function: SetFunction
    is_submodular: bool
    violation: SubmodularityViolation | None = None


__all__ = [
    "MonotonicityCheckRequest",
    "MonotonicityCheckResult",
    "MonotonicityViolation",
    "SetFunction",
    "SetFunctionEntry",
    "SetFunctionEvalRequest",
    "SetFunctionEvalResult",
    "SubmodularityCheckRequest",
    "SubmodularityCheckResult",
    "SubmodularityViolation",
]
