"""Typed wire contracts for exact bounded symbolic dynamics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.symbolic_dynamics.operations import (
    _presentation_memory,
    _require_bounded_presentation,
    _require_bounded_support,
    block_language,
    enumeration_size,
    finite_type_presentation,
    higher_block_presentation,
    normalize_forbidden_blocks,
    periodic_point_profile,
)
from jacobian.math.symbolic_dynamics.values import (
    MAX_FORBIDDEN_BLOCK_LENGTH,
    MAX_PERIOD,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
)

MAX_PERIODIC_PROFILE_DIGITS = 100_000


class FiniteTypeShiftRequest(StrictModel):
    shift: ForbiddenBlockShift

    @model_validator(mode="after")
    def require_bounded_presentation(self) -> Self:
        memory = _presentation_memory(self.shift)
        _require_bounded_presentation(self.shift, memory)
        return self


class FiniteTypeShiftResult(FiniteTypeShiftRequest):
    presentation: BlockPresentation
    normalized_forbidden_blocks: tuple[tuple[str, ...], ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DE_BRUIJN_PRESENTATION"] = "EXACT_DE_BRUIJN_PRESENTATION"

    @model_validator(mode="after")
    def bind_presentation(self) -> Self:
        if self.presentation != finite_type_presentation(
            self.shift
        ) or self.normalized_forbidden_blocks != normalize_forbidden_blocks(self.shift):
            raise ValueError("finite-type presentation is not bound to the request")
        return self


class BlockLanguageRequest(StrictModel):
    shift: ForbiddenBlockShift
    block_length: int = Field(ge=0, le=MAX_FORBIDDEN_BLOCK_LENGTH)

    @model_validator(mode="after")
    def require_bounded_enumeration(self) -> Self:
        enumeration_size(len(self.shift.alphabet), self.block_length)
        _require_bounded_support(self.shift)
        return self


class BlockLanguageResult(BlockLanguageRequest):
    allowed_blocks: tuple[tuple[str, ...], ...]
    count: int = Field(ge=0)
    complete: Literal[True] = True
    scope: Literal["ALL_OCCURRING_BLOCKS_OF_REQUESTED_LENGTH"] = (
        "ALL_OCCURRING_BLOCKS_OF_REQUESTED_LENGTH"
    )
    method: Literal["EXACT_PRESENTATION_SUPPORT_ENUMERATION"] = (
        "EXACT_PRESENTATION_SUPPORT_ENUMERATION"
    )

    @model_validator(mode="after")
    def bind_language(self) -> Self:
        expected = block_language(self.shift, self.block_length)
        if self.allowed_blocks != expected or self.count != len(expected):
            raise ValueError("block language is not bound to the request")
        return self


class PeriodicPointProfileRequest(StrictModel):
    shift: AdjacencyShift
    max_period: int = Field(ge=1, le=MAX_PERIOD)

    @model_validator(mode="after")
    def require_bounded_matrix_powering(self) -> Self:
        states = len(self.shift.matrix)
        if states**3 * self.max_period > 10_000_000:
            raise ValueError("periodic-point matrix powering exceeds the work bound")
        max_row_sum = max(sum(row) for row in self.shift.matrix)
        count_bound = states * max(1, max_row_sum) ** self.max_period
        aggregate_digits = 3 * self.max_period * len(str(count_bound))
        if aggregate_digits > MAX_PERIODIC_PROFILE_DIGITS:
            raise ValueError("periodic-point profile exceeds the output digit bound")
        return self


class PeriodicPointProfileResult(PeriodicPointProfileRequest):
    periods: tuple[int, ...]
    fixed_point_counts: tuple[CanonicalInteger, ...]
    least_period_point_counts: tuple[CanonicalInteger, ...]
    primitive_orbit_counts: tuple[CanonicalInteger, ...]
    complete_through_period: int = Field(ge=1, le=MAX_PERIOD)
    method: Literal["EXACT_MATRIX_TRACES_AND_MOBIUS_INVERSION"] = (
        "EXACT_MATRIX_TRACES_AND_MOBIUS_INVERSION"
    )

    @model_validator(mode="after")
    def bind_profile(self) -> Self:
        fixed, exact, orbits = periodic_point_profile(self.shift, self.max_period)
        if (
            self.periods != tuple(range(1, self.max_period + 1))
            or self.fixed_point_counts
            != tuple(format_canonical_integer(value) for value in fixed)
            or self.least_period_point_counts
            != tuple(format_canonical_integer(value) for value in exact)
            or self.primitive_orbit_counts
            != tuple(format_canonical_integer(value) for value in orbits)
            or self.complete_through_period != self.max_period
        ):
            raise ValueError("periodic-point profile is not bound to the request")
        return self


class HigherBlockRequest(StrictModel):
    shift: ForbiddenBlockShift
    block_length: int = Field(ge=1, le=MAX_FORBIDDEN_BLOCK_LENGTH)

    @model_validator(mode="after")
    def require_exact_bounded_presentation(self) -> Self:
        required_memory = _presentation_memory(self.shift)
        if self.block_length < required_memory:
            raise ValueError(
                "block_length must be at least the SFT presentation memory"
            )
        _require_bounded_presentation(self.shift, self.block_length)
        return self


class HigherBlockResult(HigherBlockRequest):
    presentation: BlockPresentation
    complete: Literal[True] = True
    method: Literal["EXACT_ALLOWED_OVERLAP_PRESENTATION"] = (
        "EXACT_ALLOWED_OVERLAP_PRESENTATION"
    )

    @model_validator(mode="after")
    def bind_higher_block_presentation(self) -> Self:
        if self.presentation != higher_block_presentation(
            self.shift, self.block_length
        ):
            raise ValueError("higher-block presentation is not bound to the request")
        return self


__all__ = [
    "BlockLanguageRequest",
    "BlockLanguageResult",
    "FiniteTypeShiftRequest",
    "FiniteTypeShiftResult",
    "HigherBlockRequest",
    "HigherBlockResult",
    "PeriodicPointProfileRequest",
    "PeriodicPointProfileResult",
]
