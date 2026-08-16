"""Typed wire contracts for convex analysis operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational

MAX_DIMENSION = 20
MAX_PIECES = 100


class AffinePiece(ContractModel):
    """One affine piece: f(x) = <a, x> + b."""

    piece_id: str = Field(min_length=1, max_length=64)
    coefficients: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )
    intercept: CanonicalRational


class MaxAffineFunction(ContractModel):
    """A max-affine function f(x) = max_i { <a_i, x> + b_i }."""

    pieces: tuple[AffinePiece, ...] = Field(min_length=1, max_length=MAX_PIECES)

    @model_validator(mode="after")
    def require_uniform_dimension(self) -> Self:
        dim = len(self.pieces[0].coefficients)
        for p in self.pieces[1:]:
            if len(p.coefficients) != dim:
                raise ValueError("all pieces must have the same dimension")
        ids = [p.piece_id for p in self.pieces]
        if len(ids) != len(set(ids)):
            raise ValueError("piece IDs must be unique")
        return self


class RationalPoint(ContractModel):
    """A rational point in the affine function's domain."""

    coordinates: tuple[CanonicalRational, ...] = Field(
        min_length=1, max_length=MAX_DIMENSION
    )


class MaxAffineEvalRequest(ContractModel):
    """Evaluate a max-affine function at a point."""

    function: MaxAffineFunction
    point: RationalPoint

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        dim = len(self.function.pieces[0].coefficients)
        if len(self.point.coordinates) != dim:
            raise ValueError("point dimension must match function dimension")
        return self


class MaxAffineEvalResult(ContractModel):
    """Result of max-affine evaluation: value and active pieces."""

    value: str
    active_pieces: tuple[str, ...]
    all_values: tuple[tuple[str, str], ...] = ()


class MaxAffineSubdifferentialRequest(ContractModel):
    """Compute the subdifferential at a point."""

    function: MaxAffineFunction
    point: RationalPoint

    @model_validator(mode="after")
    def require_matching_dimension(self) -> Self:
        dim = len(self.function.pieces[0].coefficients)
        if len(self.point.coordinates) != dim:
            raise ValueError("point dimension must match function dimension")
        return self


class MaxAffineSubdifferentialResult(ContractModel):
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
