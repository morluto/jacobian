"""Typed result for source degree-multiset reconstruction."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.decks.anonymous_vertex_edge_count._models import (
    MAX_ANONYMOUS_VERTEX_DECK_ORDER,
    AnonymousVertexDeckEdgeCount,
)

Degree = Annotated[int, Field(ge=0, le=MAX_ANONYMOUS_VERTEX_DECK_ORDER - 1)]


class AnonymousVertexDeckDegreeMultiset(StrictModel):
    """Reconstructed source degrees together with the edge-count derivation.

    The edge-count value retains the anonymous deck and source order. Degree
    entries are in nonincreasing order; the tuple is empty for the order-zero
    graph.
    """

    edge_count: AnonymousVertexDeckEdgeCount
    degrees: tuple[Degree, ...] = Field(max_length=MAX_ANONYMOUS_VERTEX_DECK_ORDER)

    @model_validator(mode="after")
    def require_canonical_degrees(self) -> Self:
        if len(self.degrees) != self.edge_count.source_order:
            raise ValueError("degree count must equal the source graph order")
        adjacent_degrees = zip(self.degrees, self.degrees[1:], strict=False)
        if any(left < right for left, right in adjacent_degrees):
            raise ValueError("degrees must be in nonincreasing order")
        return self
