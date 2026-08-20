"""Typed exact contracts for finite vector families and frames."""

from __future__ import annotations

from fractions import Fraction
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger, CanonicalRational
from jacobian._models import StrictModel

MAX_VECTORS, MAX_DIM, MAX_VALUE = 32, 16, 1000


class VectorFamilyRequest(StrictModel):
    """A bounded family in the standard ordered coordinate space."""

    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_VECTORS)

    @model_validator(mode="after")
    def require_rectangular_family(self) -> Self:
        dimension = len(self.vectors[0])
        if not 1 <= dimension <= MAX_DIM:
            raise ValueError(f"vector dimension must be between 1 and {MAX_DIM}")
        if any(len(vector) != dimension for vector in self.vectors):
            raise ValueError("all vectors must have equal dimension")
        if any(abs(entry) > MAX_VALUE for vector in self.vectors for entry in vector):
            raise ValueError("vector entries must be bounded")
        return self


class FiniteFrameRequest(VectorFamilyRequest):
    """A vector family spanning its full standard ambient space."""

    @model_validator(mode="after")
    def require_full_span(self) -> Self:
        from sympy import Matrix

        if Matrix(self.vectors).rank() != len(self.vectors[0]):
            raise ValueError("a finite frame must span its ambient space")
        return self


class CoherenceRequest(FiniteFrameRequest):
    @model_validator(mode="after")
    def require_nonzero_vectors(self) -> Self:
        if any(not any(vector) for vector in self.vectors):
            raise ValueError("coherence requires every vector to be nonzero")
        return self


def _dot(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a * b for a, b in zip(left, right, strict=True))


class GramResult(VectorFamilyRequest):
    gram: tuple[tuple[int, ...], ...]
    dimension: int = Field(ge=1)
    method: str = "DOT_PRODUCT"

    @model_validator(mode="after")
    def bind_gram(self) -> Self:
        expected = tuple(
            tuple(_dot(left, right) for right in self.vectors) for left in self.vectors
        )
        if self.gram != expected or self.dimension != len(self.vectors[0]):
            raise ValueError("Gram result is not bound to its vector family")
        return self


class CoherenceResult(CoherenceRequest):
    coherence_squared: CanonicalRational
    maximizing_pair: tuple[int, int] | None
    method: str = "EXACT_MAX_SQUARED_NORMALIZED_INNER_PRODUCT"

    @model_validator(mode="after")
    def bind_coherence(self) -> Self:
        candidates = []
        for left in range(len(self.vectors)):
            for right in range(left + 1, len(self.vectors)):
                dot = _dot(self.vectors[left], self.vectors[right])
                candidates.append(
                    (
                        Fraction(
                            dot * dot,
                            _dot(self.vectors[left], self.vectors[left])
                            * _dot(self.vectors[right], self.vectors[right]),
                        ),
                        (left, right),
                    )
                )
        value, pair = max(candidates, default=(Fraction(0), None))
        if (
            self.coherence_squared.as_fraction() != value
            or self.maximizing_pair != pair
        ):
            raise ValueError("coherence result is not bound to its frame")
        return self


class FramePotentialResult(FiniteFrameRequest):
    potential: CanonicalInteger
    method: str = "EXACT_GRAM_SQUARE_SUM"

    @model_validator(mode="after")
    def bind_potential(self) -> Self:
        expected = sum(
            _dot(left, right) ** 2 for left in self.vectors for right in self.vectors
        )
        if int(self.potential) != expected:
            raise ValueError("frame potential is not bound to its frame")
        return self


FrameRequest = FiniteFrameRequest

__all__ = [
    "CoherenceRequest",
    "CoherenceResult",
    "FiniteFrameRequest",
    "FramePotentialResult",
    "FrameRequest",
    "GramResult",
    "VectorFamilyRequest",
]
