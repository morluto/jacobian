"""Typed wire contracts for exact vertex-deletion deck operations."""

from __future__ import annotations

from typing import Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_DECK_VERTICES = 64
"""Admission cap on source vertices so the complete card family fits output."""

MAX_DECK_CARD_EDGES = 130_000
MAX_EDGE_DECK_EDGES = 130_000
MAX_UNLABELLED_DECK_VERTICES = 10
MAX_UNLABELLED_DECK_ISOMORPHISM_WORK = 2_000_000
"""Admission cap on aggregate card edges across the whole family."""


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph_deck.{reason}", message)


def _isomorphic(left: SimpleUndirectedGraph, right: SimpleUndirectedGraph) -> bool:
    """Check the source-labelled cards under the admitted exact quotient."""
    import networkx as nx

    first: nx.Graph[str] = nx.Graph()
    first.add_nodes_from(left.vertices)
    first.add_edges_from(left.edges)
    second: nx.Graph[str] = nx.Graph()
    second.add_nodes_from(right.vertices)
    second.add_edges_from(right.edges)
    return nx.is_isomorphic(first, second)


class VertexDeckRequest(StrictModel):
    """Compute the complete source-bound vertex-deletion family of a graph."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Simple graph with at most 64 vertices; aggregate retained card edges "
            "must fit the 130000-edge complete-family output envelope."
        )
    )


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
        deleted = tuple(card.deleted_vertex for card in self.cards)
        if deleted != self.source.vertices:
            raise _validation_error(
                "family_deletion_coverage",
                "deletion keys must cover the source vertex axis exactly once in source order",
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
            expected_edges = tuple(
                edge
                for edge in self.source.edges
                if edge[0] != card.deleted_vertex and edge[1] != card.deleted_vertex
            )
            if card.card.edges != expected_edges:
                raise _validation_error(
                    "card_deletion_relation",
                    "each card must equal the source graph with its bound vertex removed",
                )
            if any(edge not in source_edges for edge in card.card.edges):
                raise _validation_error(
                    "card_foreign_edge",
                    "card edges must be retained source edges",
                )
            if (
                len(card.card.edges) != card.retained_edge_count
                or card.retained_edge_count + card.deleted_edge_count
                != len(self.source.edges)
                or card.deleted_edge_count
                != sum(card.deleted_vertex in edge for edge in self.source.edges)
            ):
                raise _validation_error(
                    "card_edge_count",
                    "retained and deleted edge receipts must bind the source card",
                )
        expected_edge_appearances = (
            tuple(len(self.source.vertices) - 2 for _ in self.source.edges)
            if len(self.source.vertices) >= 2
            else tuple(0 for _ in self.source.edges)
        )
        expected_vertex_appearances = tuple(
            len(self.source.vertices) - 1 for _ in self.source.vertices
        )
        if self.edge_appearances != expected_edge_appearances:
            raise _validation_error(
                "family_edge_receipt",
                "edge ledgers must equal the complete source deletion counts",
            )
        if self.vertex_appearances != expected_vertex_appearances:
            raise _validation_error(
                "family_vertex_receipt",
                "vertex ledgers must equal the complete source deletion counts",
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


class EdgeDeckRequest(StrictModel):
    """Compute one card for each source edge deletion."""

    graph: SimpleUndirectedGraph = Field(
        description=(
            "Simple graph whose edge-deletion cards have aggregate output at most "
            "130000 retained edge entries (the bound is |E| * max(|E|-1, 0))."
        )
    )


class SourceBoundEdgeCard(StrictModel):
    """One edge-deleted card retaining its source edge key and vertex axis."""

    deleted_edge: tuple[str, str]
    card: SimpleUndirectedGraph
    retained_vertices: tuple[str, ...]
    retained_edge_count: int = Field(ge=0)


class EdgeDeletionFamily(StrictModel):
    """Complete source-bound edge-deletion deck with multiplicity-free cards."""

    source: SimpleUndirectedGraph
    cards: tuple[SourceBoundEdgeCard, ...]

    @model_validator(mode="after")
    def require_edge_coverage(self) -> Self:
        if len(self.cards) != len(self.source.edges):
            raise _validation_error(
                "edge_family_card_count", "one card is required per source edge"
            )
        keys = tuple(card.deleted_edge for card in self.cards)
        if keys != self.source.edges:
            raise _validation_error(
                "edge_family_coverage",
                "deleted edges must cover the source edge axis exactly once in source order",
            )
        for card in self.cards:
            if (
                card.retained_vertices != self.source.vertices
                or card.card.vertices != self.source.vertices
            ):
                raise _validation_error(
                    "edge_card_vertices",
                    "edge cards retain the complete source vertex axis",
                )
            if card.deleted_edge not in self.source.edges:
                raise _validation_error(
                    "edge_card_key", "deleted edge must belong to the source graph"
                )
            expected_edges = tuple(
                edge for edge in self.source.edges if edge != card.deleted_edge
            )
            if card.card.edges != expected_edges:
                raise _validation_error(
                    "edge_card_deletion",
                    "each card must equal the source graph with exactly its deleted edge removed",
                )
            if len(card.card.edges) != card.retained_edge_count:
                raise _validation_error(
                    "edge_card_count", "retained edge count must match the card"
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class UnlabelledDeckRequest(StrictModel):
    """Consume a serialized canonical edge-deletion family unchanged."""

    deck: EdgeDeletionFamily = Field(
        description=(
            "A complete source-bound edge deck; quotient is admitted for at most "
            "10 source vertices and 2000000 units of pairwise isomorphism work."
        )
    )


class UnlabelledDeckClass(StrictModel):
    """One isomorphism class in an unlabelled deck quotient."""

    representative: SimpleUndirectedGraph
    multiplicity: int = Field(ge=1)
    card_indices: tuple[int, ...]


class UnlabelledDeck(StrictModel):
    """The exact multiset quotient of a source-bound edge deck."""

    source: SimpleUndirectedGraph
    classes: tuple[UnlabelledDeckClass, ...]
    card_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_partition(self) -> Self:
        if len(self.source.vertices) > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "quotient_vertex_bound",
                "unlabelled deck exceeds the exact isomorphism envelope",
            )
        if (
            len(self.source.edges) * max(len(self.source.vertices), 1) ** 2
            > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK
        ):
            raise _validation_error(
                "quotient_work_bound",
                "unlabelled deck exceeds the exact isomorphism work envelope",
            )
        if (
            self.card_count != len(self.source.edges)
            or sum(item.multiplicity for item in self.classes) != self.card_count
        ):
            raise _validation_error(
                "quotient_card_count",
                "deck classes must partition every source edge card",
            )
        indices = [index for item in self.classes for index in item.card_indices]
        if sorted(indices) != list(range(self.card_count)):
            raise _validation_error(
                "quotient_indices", "deck class indices must partition the card axis"
            )
        if any(len(item.card_indices) != item.multiplicity for item in self.classes):
            raise _validation_error(
                "quotient_multiplicity",
                "multiplicity must equal class card-index count",
            )
        for left_index, left in enumerate(self.classes):
            for right in self.classes[left_index + 1 :]:
                if _isomorphic(left.representative, right.representative):
                    raise _validation_error(
                        "quotient_class_maximality",
                        "distinct quotient classes must have non-isomorphic representatives",
                    )
        for item in self.classes:
            if (
                tuple(sorted(item.card_indices)) != item.card_indices
                or not item.card_indices
                or item.representative.vertices != self.source.vertices
            ):
                raise _validation_error(
                    "quotient_source_cards",
                    "each quotient representative must bind the source card axis",
                )
            representative_edges = tuple(
                edge
                for edge in self.source.edges
                if edge != self.source.edges[item.card_indices[0]]
            )
            if item.representative.edges != representative_edges:
                raise _validation_error(
                    "quotient_source_cards",
                    "each quotient representative must be the first source card",
                )
            for card_index in item.card_indices:
                card_edges = tuple(
                    edge
                    for edge in self.source.edges
                    if edge != self.source.edges[card_index]
                )
                card = SimpleUndirectedGraph(
                    vertices=self.source.vertices,
                    edges=card_edges,
                )
                if not _isomorphic(card, item.representative):
                    raise _validation_error(
                        "quotient_isomorphism",
                        "every indexed source card must be isomorphic to its representative",
                    )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_DECK_CARD_EDGES",
    "MAX_DECK_VERTICES",
    "MAX_EDGE_DECK_EDGES",
    "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK",
    "MAX_UNLABELLED_DECK_VERTICES",
    "EdgeDeckRequest",
    "EdgeDeletionFamily",
    "SourceBoundEdgeCard",
    "SourceBoundVertexCard",
    "UnlabelledDeck",
    "UnlabelledDeckClass",
    "UnlabelledDeckRequest",
    "VertexDeckRequest",
    "VertexDeletionFamily",
]
