"""Typed source-order reconstruction from an anonymous vertex deck."""

from __future__ import annotations

from typing import Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.decks._models import (
    MAX_UNLABELLED_DECK_VERTICES,
    AnonymousGraphCardMultiset,
)

MAX_ANONYMOUS_VERTEX_SOURCE_ORDER = MAX_UNLABELLED_DECK_VERTICES + 1


class AnonymousVertexDeckSourceOrder(StrictModel):
    """The source order inferred from a complete anonymous vertex-card deck.

    The empty deck is the vertex-deletion deck of the order-zero graph. A
    nonempty deck has ``card_order + 1`` cards, counting multiplicity.
    """

    deck: AnonymousGraphCardMultiset
    source_order: StrictInt = Field(ge=0, le=MAX_ANONYMOUS_VERTEX_SOURCE_ORDER)

    @model_validator(mode="after")
    def require_complete_vertex_deck(self) -> Self:
        card_order = self.deck.card_order
        card_count = 0
        for card_class in self.deck.classes:
            card_count += card_class.multiplicity
            if card_count > MAX_ANONYMOUS_VERTEX_SOURCE_ORDER:
                raise PydanticCustomError(
                    "graph_deck.vertex_source_order_card_count",
                    "card multiplicities exceed the supported source order",
                )
        expected_order = (
            card_count if card_order == 0 and card_count in (0, 1) else card_order + 1
        )
        if card_count != expected_order or self.source_order != expected_order:
            raise PydanticCustomError(
                "graph_deck.vertex_source_order_incomplete",
                "a complete vertex deck has one card per source vertex, each of order n-1",
            )
        return self

    @classmethod
    def _from_kernel(cls, deck: AnonymousGraphCardMultiset, source_order: int) -> Self:
        """Build the result after its deck cardinality has been admitted."""
        return cls.model_construct(deck=deck, source_order=source_order)
