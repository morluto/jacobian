"""Typed wire contracts for exact bounded symbolic dynamics operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer
from jacobian.math.dynamics.symbolic.values import (
    MAX_FORBIDDEN_BLOCK_LENGTH,
    MAX_PERIOD,
    AdjacencyShift,
    BlockPresentation,
    ForbiddenBlockShift,
)
from jacobian.math.polynomials.values import RationalFunction, RationalPolynomial


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"symbolic_dynamics.{reason}", message)


class FiniteTypeShiftRequest(StrictModel):
    shift: ForbiddenBlockShift


class FiniteTypeShiftResult(FiniteTypeShiftRequest):
    presentation: BlockPresentation
    normalized_forbidden_blocks: tuple[tuple[str, ...], ...]


class BlockLanguageRequest(StrictModel):
    shift: ForbiddenBlockShift
    block_length: int = Field(ge=0, le=MAX_FORBIDDEN_BLOCK_LENGTH)


class BlockLanguageResult(BlockLanguageRequest):
    allowed_blocks: tuple[tuple[str, ...], ...]
    count: int = Field(ge=0)
    scope: Literal["ALL_OCCURRING_BLOCKS_OF_REQUESTED_LENGTH"] = (
        "ALL_OCCURRING_BLOCKS_OF_REQUESTED_LENGTH"
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


class PeriodicPointProfileResult(PeriodicPointProfileRequest):
    periods: tuple[int, ...]
    fixed_point_counts: tuple[CanonicalInteger, ...]
    least_period_point_counts: tuple[CanonicalInteger, ...]
    primitive_orbit_counts: tuple[CanonicalInteger, ...]
    complete_through_period: int = Field(ge=1, le=MAX_PERIOD)

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


class HigherBlockResult(HigherBlockRequest):
    presentation: BlockPresentation


class ArtinMazurZetaRequest(StrictModel):
    """Request the exact Artin--Mazur zeta function of one edge shift."""

    shift: AdjacencyShift


class ArtinMazurZetaResult(ArtinMazurZetaRequest):
    determinant_polynomial: RationalPolynomial
    zeta_function: RationalFunction
    convention: Literal["EDGE_SHIFT_ARTIN_MAZUR_ZETA"] = "EDGE_SHIFT_ARTIN_MAZUR_ZETA"


def _from_kernel_artin_mazur_zeta(
    request: ArtinMazurZetaRequest,
    determinant_polynomial: RationalPolynomial,
    zeta_function: RationalFunction,
) -> ArtinMazurZetaResult:
    """Construct a zeta result from the trusted one-pass owner kernel."""

    if determinant_polynomial.variables != ("t",) or zeta_function.variables != ("t",):
        raise _validation_error(
            "artin_mazur_zeta_computed_shape",
            "computed zeta values must retain the canonical t axis",
        )
    return ArtinMazurZetaResult(
        shift=request.shift,
        determinant_polynomial=determinant_polynomial,
        zeta_function=zeta_function,
        convention="EDGE_SHIFT_ARTIN_MAZUR_ZETA",
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
