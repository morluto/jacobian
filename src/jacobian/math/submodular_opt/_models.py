"""Typed wire contracts for submodular optimization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

# The checks run the local characterizations: monotonicity scans n*2^n
# covering relations and submodularity C(n,2)*2^n local pairs, both with
# early exit.  The absolute ceiling is transport-derived: the request
# carries a complete 2^n-entry table whose serialized size grows as
# 2^n * (1.5n + 25) bytes, so 9 MiB of canonical input budget admits
# n <= 17; 16 keeps the worst-case full-scan work (~8M exact pair checks)
# comfortably inside the synchronous envelope.  The per-request byte
# preflight below is the binding, result-sensitive guard.
#
# Scan work also scales with coefficient height: every inequality subtracts
# and compares exact Fractions.  The two check requests therefore bound each
# value to 128 numerator/denominator digits, keeping every one of the ~8M
# comparisons on small big-ints so the documented seconds-scale envelope
# holds.  This is scan-specific admission: the shared entry type stays at
# the canonical ceiling so the single-lookup evaluator can return any exact
# representable value.
MAX_GROUND_SET = 16
_MAX_TABLE_WIRE_BYTES = 9 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 25
MAX_SUBMODULAR_SCAN_VALUE_DIGITS = 128


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by submodular-opt contracts."""

    return PydanticCustomError(f"submodular_opt.{reason}", message)


def _require_scan_value_height(function: SetFunction) -> None:
    """Bound value heights for the millions-of-comparisons scan kernels."""

    for entry in function.entries:
        try:
            require_bounded_rational(
                entry.value,
                max_digits=MAX_SUBMODULAR_SCAN_VALUE_DIGITS,
                label="set-function scan value",
            )
        except ValueError as error:
            raise _validation_error("scan_value_height_exceeded", str(error)) from error


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
        estimated_bytes = sum(
            _ENTRY_OVERHEAD_BYTES
            + len(entry.value.num)
            + len(entry.value.den)
            + 2 * sum(len(str(element)) + 1 for element in entry.subset)
            for entry in self.entries
        )
        if estimated_bytes > _MAX_TABLE_WIRE_BYTES:
            raise _validation_error(
                "table_transport_envelope_exceeded",
                "set-function table exceeds the "
                f"{_MAX_TABLE_WIRE_BYTES}-byte transport envelope; the complete "
                "2^n table cannot fit at this ground-set size",
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
    """Value of f(S) or unknown."""

    value: str
    found: bool


class MonotonicityCheckRequest(StrictModel):
    """Check if a set function is monotone non-decreasing."""

    function: SetFunction

    @model_validator(mode="after")
    def require_bounded_scan_height(self) -> Self:
        _require_scan_value_height(self.function)
        return self


class MonotonicityCheckResult(StrictModel):
    """Whether the function is monotone non-decreasing."""

    is_monotone: bool
    violation: str


class SubmodularityCheckRequest(StrictModel):
    """Check if a set function is submodular."""

    function: SetFunction

    @model_validator(mode="after")
    def require_bounded_scan_height(self) -> Self:
        _require_scan_value_height(self.function)
        return self


class SubmodularityCheckResult(StrictModel):
    """Whether the function is submodular."""

    is_submodular: bool
    violation: str


__all__ = [
    "MonotonicityCheckRequest",
    "MonotonicityCheckResult",
    "SetFunction",
    "SetFunctionEntry",
    "SetFunctionEvalRequest",
    "SetFunctionEvalResult",
    "SubmodularityCheckRequest",
    "SubmodularityCheckResult",
]
