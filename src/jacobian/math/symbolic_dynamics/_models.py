"""Typed wire contracts for exact bounded symbolic dynamics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.polynomials.values import RationalFunction, RationalPolynomial
from jacobian.math.symbolic_dynamics._bounds import (
    MAX_ZETA_REPLAY_PERIOD,
    enumeration_size,
    presentation_memory,
    require_bounded_presentation,
    require_bounded_support,
    require_zeta_budget,
)
from jacobian.math.symbolic_dynamics.values import (
    MAX_FORBIDDEN_BLOCK_LENGTH,
    MAX_PERIOD,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
)

MAX_PERIODIC_PROFILE_DIGITS = 100_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"symbolic_dynamics.{reason}", message)


def _validation_error_from_message(error: ValueError) -> PydanticCustomError:
    reasons = {
        "presentation adjacency exceeds the result bound": "finite_type_presentation_result_bound",
        "requested block enumeration exceeds the work bound": "block_language_work_bound",
        "zeta polynomial exceeds the coefficient digit bound": "zeta_coefficient_digit_bound",
        "zeta determinant and replay exceed the work bound": "zeta_work_bound",
        "zeta result exceeds the aggregate digit bound": "zeta_result_digit_bound",
    }
    return _validation_error(reasons.get(str(error), "request_bound"), str(error))


class FiniteTypeShiftRequest(StrictModel):
    shift: ForbiddenBlockShift


class FiniteTypeShiftResult(FiniteTypeShiftRequest):
    presentation: BlockPresentation
    normalized_forbidden_blocks: tuple[tuple[str, ...], ...]
    complete: Literal[True] = True
    method: Literal["EXACT_DE_BRUIJN_PRESENTATION"] = "EXACT_DE_BRUIJN_PRESENTATION"


class BlockLanguageRequest(StrictModel):
    shift: ForbiddenBlockShift
    block_length: int = Field(ge=0, le=MAX_FORBIDDEN_BLOCK_LENGTH)

    @model_validator(mode="after")
    def require_bounded_enumeration(self) -> Self:
        try:
            enumeration_size(len(self.shift.alphabet), self.block_length)
            require_bounded_support(self.shift)
        except ValueError as error:
            raise _validation_error_from_message(error) from error
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
    def require_language_count(self) -> Self:
        if self.count != len(self.allowed_blocks):
            raise _validation_error(
                "block_language_count", "block-language count must match its rows"
            )
        return self


class PeriodicPointProfileRequest(StrictModel):
    shift: AdjacencyShift
    max_period: int = Field(ge=1, le=MAX_PERIOD)

    @model_validator(mode="after")
    def require_bounded_matrix_powering(self) -> Self:
        states = len(self.shift.matrix)
        if states**3 * self.max_period > 10_000_000:
            raise _validation_error(
                "periodic_point_work_bound",
                "periodic-point matrix powering exceeds the work bound",
            )
        max_row_sum = max(sum(row) for row in self.shift.matrix)
        count_bound = states * max(1, max_row_sum) ** self.max_period
        aggregate_digits = 3 * self.max_period * len(str(count_bound))
        if aggregate_digits > MAX_PERIODIC_PROFILE_DIGITS:
            raise _validation_error(
                "periodic_point_result_bound",
                "periodic-point profile exceeds the output digit bound",
            )
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
    def require_profile_shape(self) -> Self:
        if (
            self.periods != tuple(range(1, self.max_period + 1))
            or len(self.fixed_point_counts) != self.max_period
            or len(self.least_period_point_counts) != self.max_period
            or len(self.primitive_orbit_counts) != self.max_period
            or self.complete_through_period != self.max_period
        ):
            raise _validation_error(
                "periodic_point_profile_shape",
                "periodic-point profile must cover the requested period range",
            )
        return self


class HigherBlockRequest(StrictModel):
    shift: ForbiddenBlockShift
    block_length: int = Field(ge=1, le=MAX_FORBIDDEN_BLOCK_LENGTH)

    @model_validator(mode="after")
    def require_exact_bounded_presentation(self) -> Self:
        required_memory = presentation_memory(self.shift)
        if self.block_length < required_memory:
            raise _validation_error(
                "higher_block_memory_bound",
                "block_length must be at least the SFT presentation memory",
            )
        try:
            require_bounded_presentation(self.shift, self.block_length)
        except ValueError as error:
            raise _validation_error_from_message(error) from error
        return self


class HigherBlockResult(HigherBlockRequest):
    presentation: BlockPresentation
    complete: Literal[True] = True
    method: Literal["EXACT_ALLOWED_OVERLAP_PRESENTATION"] = (
        "EXACT_ALLOWED_OVERLAP_PRESENTATION"
    )


class ArtinMazurZetaRequest(StrictModel):
    """Request the exact Artin--Mazur zeta function of one edge shift."""

    shift: AdjacencyShift
    replay_period: int = Field(ge=1, le=MAX_ZETA_REPLAY_PERIOD)

    @model_validator(mode="after")
    def require_bounded_exact_zeta(self) -> Self:
        try:
            require_zeta_budget(self.shift, self.replay_period)
        except ValueError as error:
            raise _validation_error_from_message(error) from error
        return self


class ArtinMazurZetaReplayRow(StrictModel):
    period: int = Field(ge=1, le=MAX_ZETA_REPLAY_PERIOD)
    trace_fixed_points: CanonicalInteger
    logarithmic_derivative_coefficient: CanonicalInteger


class ArtinMazurZetaResult(ArtinMazurZetaRequest):
    determinant_polynomial: RationalPolynomial
    zeta_function: RationalFunction
    replay: tuple[ArtinMazurZetaReplayRow, ...] = Field(
        min_length=1, max_length=MAX_ZETA_REPLAY_PERIOD
    )
    convention: Literal["EDGE_SHIFT_ARTIN_MAZUR_ZETA"] = "EDGE_SHIFT_ARTIN_MAZUR_ZETA"
    method: Literal["SYMPY_EXACT_CHARACTERISTIC_POLYNOMIAL"] = (
        "SYMPY_EXACT_CHARACTERISTIC_POLYNOMIAL"
    )

    @model_validator(mode="after")
    def require_replay_scope(self) -> Self:
        if len(self.replay) != self.replay_period or tuple(
            row.period for row in self.replay
        ) != tuple(range(1, self.replay_period + 1)):
            raise _validation_error(
                "artin_mazur_zeta_replay_scope",
                "zeta replay must cover each requested period exactly once",
            )
        return self


def _from_kernel_artin_mazur_zeta(
    request: ArtinMazurZetaRequest,
    determinant_polynomial: RationalPolynomial,
    zeta_function: RationalFunction,
    coefficients: tuple[int, ...],
) -> ArtinMazurZetaResult:
    """Construct a zeta result from the trusted one-pass owner kernel."""

    if (
        determinant_polynomial.variables != ("t",)
        or zeta_function.variables != ("t",)
        or len(coefficients) != request.replay_period
    ):
        raise _validation_error(
            "artin_mazur_zeta_computed_shape",
            "computed zeta values must retain the canonical t axis and replay scope",
        )
    return ArtinMazurZetaResult(
        shift=request.shift,
        replay_period=request.replay_period,
        determinant_polynomial=determinant_polynomial,
        zeta_function=zeta_function,
        replay=tuple(
            ArtinMazurZetaReplayRow(
                period=period,
                trace_fixed_points=format_canonical_integer(coefficient),
                logarithmic_derivative_coefficient=format_canonical_integer(
                    coefficient
                ),
            )
            for period, coefficient in enumerate(coefficients, 1)
        ),
        convention="EDGE_SHIFT_ARTIN_MAZUR_ZETA",
        method="SYMPY_EXACT_CHARACTERISTIC_POLYNOMIAL",
    )


def _from_kernel_finite_type_shift(
    request: FiniteTypeShiftRequest,
    presentation: BlockPresentation,
    normalized_forbidden_blocks: tuple[tuple[str, ...], ...],
) -> FiniteTypeShiftResult:
    """Construct a presentation result from its trusted owner kernel."""

    return FiniteTypeShiftResult.model_construct(
        shift=request.shift,
        presentation=presentation,
        normalized_forbidden_blocks=normalized_forbidden_blocks,
    )


def _from_kernel_block_language(
    request: BlockLanguageRequest, allowed_blocks: tuple[tuple[str, ...], ...]
) -> BlockLanguageResult:
    """Construct an enumerated language result from its trusted owner kernel."""

    return BlockLanguageResult(
        **request.model_dump(), allowed_blocks=allowed_blocks, count=len(allowed_blocks)
    )


def _from_kernel_periodic_point_profile(
    request: PeriodicPointProfileRequest,
    fixed: tuple[int, ...],
    exact: tuple[int, ...],
    orbits: tuple[int, ...],
) -> PeriodicPointProfileResult:
    """Construct a periodic profile from the trusted trace-and-inversion pass."""

    return PeriodicPointProfileResult(
        **request.model_dump(),
        periods=tuple(range(1, request.max_period + 1)),
        fixed_point_counts=tuple(format_canonical_integer(value) for value in fixed),
        least_period_point_counts=tuple(
            format_canonical_integer(value) for value in exact
        ),
        primitive_orbit_counts=tuple(
            format_canonical_integer(value) for value in orbits
        ),
        complete_through_period=request.max_period,
    )


def _from_kernel_higher_block(
    request: HigherBlockRequest, presentation: BlockPresentation
) -> HigherBlockResult:
    """Construct a higher-block result from the trusted overlap kernel."""

    return HigherBlockResult(**request.model_dump(), presentation=presentation)


__all__ = [
    "ArtinMazurZetaReplayRow",
    "ArtinMazurZetaRequest",
    "ArtinMazurZetaResult",
    "BlockLanguageRequest",
    "BlockLanguageResult",
    "FiniteTypeShiftRequest",
    "FiniteTypeShiftResult",
    "HigherBlockRequest",
    "HigherBlockResult",
    "PeriodicPointProfileRequest",
    "PeriodicPointProfileResult",
]
