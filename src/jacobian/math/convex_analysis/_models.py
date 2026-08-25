"""Typed wire contracts for convex analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits

MAX_DIMENSION = 100
MAX_PIECES = 10_000
_MAX_CONVEX_WORK_CELLS = 1_000_000
_MAX_CONVEX_WIRE_BYTES = CanonicalLimits().max_output_bytes


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    """Build a stable validation error owned by convex-analysis contracts."""

    return PydanticCustomError(f"convex_analysis.{reason}", message)


class AffinePiece(StrictModel):
    """One affine piece: f(x) = <a, x> + b."""

    piece_id: str = Field(min_length=1, max_length=64)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )
    intercept: CanonicalRational


class MaxAffineFunction(StrictModel):
    """A max-affine function f(x) = max_i { <a_i, x> + b_i }."""

    pieces: tuple[AffinePiece, ...] = Field(min_length=1, max_length=MAX_PIECES)

    @model_validator(mode="after")
    def require_uniform_dimension(self) -> Self:
        dim = len(self.pieces[0].coefficients)
        for p in self.pieces[1:]:
            if len(p.coefficients) != dim:
                raise _validation_error(
                    "piece_dimension_mismatch",
                    "all pieces must have the same dimension",
                )
        ids = [p.piece_id for p in self.pieces]
        if len(ids) != len(set(ids)):
            raise _validation_error("duplicate_piece_id", "piece IDs must be unique")
        work_cells = len(self.pieces) * dim
        if work_cells > _MAX_CONVEX_WORK_CELLS:
            raise _validation_error(
                "work_bound_exceeded",
                f"max-affine work p·d={work_cells} exceeds the "
                f"{_MAX_CONVEX_WORK_CELLS}-cell result-sensitive bound; "
                "partition the function and compose",
            )
        # Transport-sensitive: p·d small but digits huge could still exceed 10MiB
        estimated = work_cells * 16 + len(self.pieces) * 64
        if estimated > _MAX_CONVEX_WIRE_BYTES:
            raise _validation_error(
                "wire_bound_exceeded",
                "max-affine function exceeds the "
                f"{_MAX_CONVEX_WIRE_BYTES}-byte transport envelope",
            )
        return self


class RationalPoint(StrictModel):
    """A rational point in the affine function's domain."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )


class MaxAffineEvalRequest(StrictModel):
    """Evaluate a max-affine function at a point."""

    function: MaxAffineFunction
    point: RationalPoint

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        dim = len(self.function.pieces[0].coefficients)
        if len(self.point.coordinates) != dim:
            raise _validation_error(
                "point_dimension_mismatch",
                "point dimension must match function dimension",
            )
        return self


class MaxAffineEvalResult(StrictModel):
    """Result of max-affine evaluation: value and active pieces."""

    value: str
    active_pieces: tuple[str, ...]
    all_values: tuple[tuple[str, str], ...] = ()


class MaxAffineSubdifferentialRequest(StrictModel):
    """Compute the subdifferential at a point."""

    function: MaxAffineFunction
    point: RationalPoint

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        dim = len(self.function.pieces[0].coefficients)
        if len(self.point.coordinates) != dim:
            raise _validation_error(
                "point_dimension_mismatch",
                "point dimension must match function dimension",
            )
        return self


class MaxAffineSubdifferentialResult(StrictModel):
    """Subdifferential: set of active gradients (one per active piece)."""

    active_gradients: tuple[tuple[CanonicalRational, ...], ...]


__all__ = [
    "AffinePiece",
    "MaxAffineEvalRequest",
    "MaxAffineEvalResult",
    "MaxAffineFunction",
    "MaxAffineSubdifferentialRequest",
    "MaxAffineSubdifferentialResult",
    "RationalPoint",
]
