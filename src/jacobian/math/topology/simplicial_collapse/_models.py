"""Typed request and result values for one elementary simplicial collapse."""

from __future__ import annotations

from pydantic import Field

from jacobian._models import StrictModel
from jacobian.math.topology._models import (
    MAX_TOPOLOGY_DIMENSION,
    FiniteSimplicialComplex,
    SimplicialComplexRequest,
    VertexLabel,
)


class ElementaryCollapsePair(StrictModel):
    """A codimension-one face and its proposed coface."""

    face: tuple[VertexLabel, ...] = Field(
        min_length=1, max_length=MAX_TOPOLOGY_DIMENSION
    )
    coface: tuple[VertexLabel, ...] = Field(
        min_length=2, max_length=MAX_TOPOLOGY_DIMENSION + 1
    )


class ElementaryCollapseRequest(StrictModel):
    """Collapse one caller-selected free face/coface pair."""

    complex: SimplicialComplexRequest
    pair: ElementaryCollapsePair


class ElementaryCollapseResult(StrictModel):
    """Canonical source and target complexes with the removed pair."""

    source: FiniteSimplicialComplex
    target: FiniteSimplicialComplex
    removed_pair: ElementaryCollapsePair


__all__ = [
    "ElementaryCollapsePair",
    "ElementaryCollapseRequest",
    "ElementaryCollapseResult",
]
