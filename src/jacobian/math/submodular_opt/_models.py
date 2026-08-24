"""Typed wire contracts for submodular optimization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

# The checks run the local characterizations: monotonicity scans n*2^n
# covering relations and submodularity C(n,2)*2^n local pairs, both with
# early exit.  The absolute ceiling is transport-derived: the request
# carries a complete 2^n-entry table whose serialized size grows as
# 2^n * (1.5n + 25) bytes, so 9 MiB of canonical input budget admits
# n <= 17; 16 keeps the worst-case full-scan work (~8M exact pair checks)
# comfortably inside the synchronous envelope.  The per-request byte
# preflight below is the binding, result-sensitive guard.
MAX_GROUND_SET = 16
_MAX_TABLE_WIRE_BYTES = 9 * 1024 * 1024
_ENTRY_OVERHEAD_BYTES = 25


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
            raise ValueError(
                "set function table must contain exactly one value per subset"
            )
        seen: set[tuple[int, ...]] = set()
        for entry in self.entries:
            if len(entry.subset) != len(set(entry.subset)):
                raise ValueError("subset elements must be unique")
            for elem in entry.subset:
                if not (0 <= elem < self.ground_set_size):
                    raise ValueError("subset elements must be in 0..ground_set_size-1")
            key = tuple(sorted(entry.subset))
            if key in seen:
                raise ValueError("set function subsets must be unique")
            seen.add(key)
        if len(seen) != expected:
            raise ValueError(
                "set function table must contain every subset of the ground set"
            )
        estimated_bytes = sum(
            _ENTRY_OVERHEAD_BYTES
            + len(entry.value.num)
            + len(entry.value.den)
            + 2 * sum(len(str(element)) + 1 for element in entry.subset)
            for entry in self.entries
        )
        if estimated_bytes > _MAX_TABLE_WIRE_BYTES:
            raise ValueError(
                "set-function table exceeds the "
                f"{_MAX_TABLE_WIRE_BYTES}-byte transport envelope; the complete "
                "2^n table cannot fit at this ground-set size"
            )
        return self


class SetFunctionEvalRequest(StrictModel):
    """Evaluate a set function at a given subset."""

    function: SetFunction
    subset: tuple[int, ...] = Field(default=())

    @model_validator(mode="after")
    def require_valid_subset(self) -> Self:
        if len(self.subset) != len(set(self.subset)):
            raise ValueError("subset elements must be unique")
        for elem in self.subset:
            if not (0 <= elem < self.function.ground_set_size):
                raise ValueError("subset elements must be in 0..ground_set_size-1")
        return self


class SetFunctionEvalResult(StrictModel):
    """Value of f(S) or unknown."""

    value: str
    found: bool


class MonotonicityCheckRequest(StrictModel):
    """Check if a set function is monotone non-decreasing."""

    function: SetFunction


class MonotonicityCheckResult(StrictModel):
    """Whether the function is monotone non-decreasing."""

    is_monotone: bool
    violation: str


class SubmodularityCheckRequest(StrictModel):
    """Check if a set function is submodular."""

    function: SetFunction


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
