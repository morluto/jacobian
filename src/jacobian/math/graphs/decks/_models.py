"""Typed wire contracts for exact vertex-deletion deck operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_DECK_VERTICES = 64
"""Admission cap on source vertices so the complete card family fits output."""

MAX_DECK_CARD_EDGES = 130_000
"""Admission cap on aggregate card edges across the whole family."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph_deck.{reason}", message)


class VertexDeckRequest(StrictModel):
    """Compute the complete source-bound vertex-deletion family of a graph."""

    graph: SimpleUndirectedGraph


class SourceBoundVertexCard(StrictModel):
    """One exact vertex-deletion event bound to its source graph."""

    deleted_vertex: str = Field(
        min_length=1,
        description="The source vertex deleted to form this card.",
    )
    card: SimpleUndirectedGraph = Field(
        description=(
            "The induced subgraph on the retained source vertices, keeping "
            "source labels; the deleted label must not remain."
        ),
    )
    retained_vertices: tuple[str, ...] = Field(
        description=(
            "Source vertices retained in this card, in source order; the "
            "source-to-card vertex injection."
        ),
    )
    retained_edge_count: int = Field(
        ge=0,
        description="Number of source edges retained in this card.",
    )
    deleted_edge_count: int = Field(
        ge=0,
        description="Number of source edges incident to the deleted vertex.",
    )


class VertexDeletionFamily(StrictModel):
    """The complete source-bound vertex-deletion family of a finite graph.

    One card per source vertex, each equal to direct deletion of its bound
    vertex. Deserialization establishes only the retained source and bounded
    canonical shape. Kernel output uses ``_from_kernel`` after its trusted
    bounded computation and counting replay.
    """

    source: SimpleUndirectedGraph
    cards: tuple[SourceBoundVertexCard, ...]
    edge_appearances: tuple[int, ...] = Field(
        description=(
            "Per-source-edge count of cards containing that edge, aligned "
            "with `source.edges`; exactly n-2 per edge when n >= 2."
        ),
    )
    vertex_appearances: tuple[int, ...] = Field(
        description=(
            "Per-source-vertex count of card domains containing that vertex, "
            "aligned with `source.vertices`; exactly n-1 per vertex."
        ),
    )

    @model_validator(mode="after")
    def require_canonical_family(self) -> Self:
        order = len(self.source.vertices)
        if len(self.cards) != order:
            raise _validation_error(
                "family_card_count",
                "the family must hold exactly one card per source vertex",
            )
        if len(self.edge_appearances) != len(self.source.edges):
            raise _validation_error(
                "family_edge_accounting",
                "edge appearances must align with the source edge axis",
            )
        if len(self.vertex_appearances) != order:
            raise _validation_error(
                "family_vertex_accounting",
                "vertex appearances must align with the source vertex axis",
            )
        deleted = [card.deleted_vertex for card in self.cards]
        if sorted(deleted) != sorted(self.source.vertices):
            raise _validation_error(
                "family_deletion_coverage",
                "deletion keys must cover the source vertex domain exactly once",
            )
        source_edges = set(self.source.edges)
        for card in self.cards:
            if card.deleted_vertex not in self.source.vertices:
                raise _validation_error(
                    "card_deleted_vertex",
                    "deleted vertex must belong to the source graph",
                )
            expected_retained = tuple(
                vertex
                for vertex in self.source.vertices
                if vertex != card.deleted_vertex
            )
            if card.retained_vertices != expected_retained:
                raise _validation_error(
                    "card_retained_vertices",
                    "retained vertices must be the source domain minus the "
                    "deleted vertex, in source order",
                )
            if card.card.vertices != expected_retained:
                raise _validation_error(
                    "card_vertex_domain",
                    "card vertex domain must equal the retained source vertices",
                )
            if card.deleted_vertex in card.card.vertices:
                raise _validation_error(
                    "card_dangling_label",
                    "the deleted label must not remain in the card",
                )
            if any(edge not in source_edges for edge in card.card.edges):
                raise _validation_error(
                    "card_foreign_edge",
                    "card edges must be retained source edges",
                )
            if len(card.card.edges) != card.retained_edge_count:
                raise _validation_error(
                    "card_edge_count",
                    "retained edge count must equal the card edge count",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        source: SimpleUndirectedGraph,
        cards: tuple[SourceBoundVertexCard, ...],
        edge_appearances: tuple[int, ...],
        vertex_appearances: tuple[int, ...],
    ) -> Self:
        """Construct trusted output of the owner-local deletion kernel."""

        return cls.model_construct(
            source=source,
            cards=cards,
            edge_appearances=edge_appearances,
            vertex_appearances=vertex_appearances,
        )


__all__ = [
    "MAX_DECK_CARD_EDGES",
    "MAX_DECK_VERTICES",
    "SourceBoundVertexCard",
    "VertexDeckRequest",
    "VertexDeletionFamily",
]
