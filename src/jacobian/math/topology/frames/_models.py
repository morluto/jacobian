"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from typing import Self

from pydantic import Field

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian.math.topology.frames.values import (
    MAX_DIM,
    MAX_VECTOR_CELLS,
    VectorFamily,
)


class VectorFamilyRequest(VectorFamily):
    """A bounded family in the standard ordered coordinate space."""


class FiniteFrameRequest(VectorFamilyRequest):
    """Wire request for an operation whose input must span its ambient space."""


class CoherenceRequest(FiniteFrameRequest):
    """Wire request for the normalized pairwise coherence operation."""


class GramResult(VectorFamilyRequest):
    gram: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        gram: tuple[tuple[int, ...], ...],
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            gram=gram,
            dimension=len(vectors[0]),
        )


class CoherenceResult(CoherenceRequest):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None

    @classmethod
    def _from_kernel(
        cls,
        *,
        vectors: tuple[tuple[int, ...], ...],
        coherence_squared: CanonicalRational,
        maximizing_pair: tuple[int, int] | None,
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            coherence_squared=coherence_squared,
            maximizing_pair=maximizing_pair,
        )


class FramePotentialResult(FiniteFrameRequest):
    potential: CanonicalInteger

    @classmethod
    def _from_kernel(
        cls, *, vectors: tuple[tuple[int, ...], ...], potential: CanonicalInteger
    ) -> Self:
        return cls.model_construct(
            vectors=vectors,
            potential=potential,
        )


__all__ = [
    "MAX_DIM",
    "MAX_VECTOR_CELLS",
    "CoherenceRequest",
    "CoherenceResult",
    "FiniteFrameRequest",
    "FramePotentialResult",
    "GramResult",
    "VectorFamilyRequest",
]
