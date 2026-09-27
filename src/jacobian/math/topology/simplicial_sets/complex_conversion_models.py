"""Contracts for the finite-complex to simplicial-set prefix transform."""

from __future__ import annotations

from pydantic import Field, StrictInt

from jacobian._models import StrictModel
from jacobian.math.topology._models import FiniteSimplicialComplex
from jacobian.math.topology.simplicial_sets._models import (
    MAX_SIMPLICIAL_SET_DEGREE,
    FiniteTruncatedSimplicialSet,
)


class SimplicialComplexPrefixRequest(StrictModel):
    """Build the finite simplicial-set prefix associated to a complex."""

    complex: FiniteSimplicialComplex
    max_degree: StrictInt = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)


class ComplexFaceSimplexIndex(StrictModel):
    """Index of one source face as a nondegenerate target simplex."""

    dimension: StrictInt = Field(ge=0, le=MAX_SIMPLICIAL_SET_DEGREE)
    face_index: StrictInt = Field(ge=0)
    simplex_index: StrictInt = Field(ge=0)


class SimplicialComplexPrefixResult(StrictModel):
    """Associated prefix plus the source-face to nondegenerate-axis map."""

    source_complex: FiniteSimplicialComplex
    simplicial_set: FiniteTruncatedSimplicialSet
    face_simplex_indices: tuple[ComplexFaceSimplexIndex, ...] = Field(max_length=2048)


__all__ = [
    "ComplexFaceSimplexIndex",
    "SimplicialComplexPrefixRequest",
    "SimplicialComplexPrefixResult",
]
