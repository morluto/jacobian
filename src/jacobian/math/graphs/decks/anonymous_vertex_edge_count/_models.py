"""Typed result for edge-count reconstruction from an anonymous vertex deck."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.graphs.decks._models import AnonymousGraphCardMultiset

MAX_ANONYMOUS_VERTEX_DECK_ORDER = 8
MAX_ANONYMOUS_VERTEX_DECK_CARD_EDGES = MAX_ANONYMOUS_VERTEX_DECK_ORDER * (
    (MAX_ANONYMOUS_VERTEX_DECK_ORDER - 1) * (MAX_ANONYMOUS_VERTEX_DECK_ORDER - 2) // 2
)


class AnonymousVertexDeckEdgeCount(StrictModel):
    """Edge-count quotient while retaining the exact anonymous input deck.

    Orders zero and one use the unique empty-edge graph convention and have no
    divisor. Order two is excluded because its two vertex cards are identical
    for both possible source edge counts.
    """

    deck: AnonymousGraphCardMultiset
    source_order: int = Field(ge=0, le=MAX_ANONYMOUS_VERTEX_DECK_ORDER)
    card_edge_total: int = Field(ge=0, le=MAX_ANONYMOUS_VERTEX_DECK_CARD_EDGES)
    overcount_divisor: int | None = Field(
        default=None, ge=1, le=MAX_ANONYMOUS_VERTEX_DECK_ORDER - 2
    )
    source_edge_count: int = Field(
        ge=0,
        le=MAX_ANONYMOUS_VERTEX_DECK_ORDER * (MAX_ANONYMOUS_VERTEX_DECK_ORDER - 1) // 2,
    )

    @model_validator(mode="after")
    def require_edge_count_structure(self) -> Self:
        order = self.source_order
        if order in (0, 1):
            expected_card_count = order
            if (
                self.deck.card_order != 0
                or sum(item.multiplicity for item in self.deck.classes)
                != expected_card_count
                or any(item.representative.edges for item in self.deck.classes)
                or self.card_edge_total != 0
                or self.overcount_divisor is not None
                or self.source_edge_count != 0
            ):
                raise ValueError("orders zero and one have no edges or Kelly divisor")
            return self
        if order < 3:
            raise ValueError(
                "anonymous vertex-deck edge count is unsupported at order two"
            )
        if self.deck.card_order != order - 1:
            raise ValueError("deck card order must be source order minus one")
        if sum(item.multiplicity for item in self.deck.classes) != order:
            raise ValueError("a vertex deck must contain one card per source vertex")
        divisor = order - 2
        if self.overcount_divisor != divisor:
            raise ValueError("overcount divisor must equal source order minus two")
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)
