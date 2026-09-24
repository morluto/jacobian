"""Typed wire contracts for exact vertex-deletion deck operations."""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb, factorial
from typing import Annotated, Any, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel
from jacobian.math.graphs.patterns._models import (
    MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_DECK_VERTICES = 64
"""Admission cap on source vertices so the complete card family fits output."""

MAX_DECK_CARD_EDGES = 130_000
MAX_EDGE_DECK_EDGES = 130_000
MAX_UNLABELLED_DECK_VERTICES = 10
MAX_UNLABELLED_DECK_ISOMORPHISM_WORK = 2_000_000
MAX_UNLABELLED_EDGE_DECK_RESULT_BYTES = 1_000_000
MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK = 2_000_000
MAX_ANONYMOUS_CARD_RESULT_BYTES = 1_000_000
MAX_ANONYMOUS_CARD_CLASSES = MAX_ANONYMOUS_CARD_RESULT_BYTES // 64
"""Admission cap on aggregate card edges across the whole family."""
MAX_VERTEX_DECK_SOURCE_EDGES = comb(MAX_UNLABELLED_DECK_VERTICES, 2)
MAX_VERTEX_DECK_CARD_EDGE_TOTAL = MAX_UNLABELLED_DECK_VERTICES * comb(
    MAX_UNLABELLED_DECK_VERTICES - 1, 2
)

MAX_KELLY_DECK_TOTAL_WORK = MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS
MAX_KELLY_COUNT_DIGITS = 3
MAX_KELLY_SUBGRAPH_COUNT_DIGITS = 12
MAX_KELLY_RESULT_BYTES = 1_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"graph_deck.{reason}", message)


def _anonymous_canonicalization_work(order: int, card_count: int) -> int:
    """Charge each permutation for vector generation and worst-case comparison."""
    pair_count = comb(order, 2)
    return card_count * factorial(order) * (order + 2 * max(1, pair_count))


def _canonical_card_edges(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    """Return the least fixed-axis adjacency encoding in the permutation orbit."""
    n = len(vertices)
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edge_indices = {frozenset((index[left], index[right])) for left, right in edges}
    pairs = tuple(combinations(range(n), 2))
    best: tuple[int, ...] | None = None
    for order in permutations(range(n)):
        bits = tuple(
            int(frozenset((order[i], order[j])) in edge_indices) for i, j in pairs
        )
        if best is None or bits < best:
            best = bits
    assert best is not None
    labels = tuple(f"v{i:02d}" for i in range(n))
    return tuple(
        (labels[i], labels[j]) for bit, (i, j) in zip(best, pairs, strict=True) if bit
    )


class AnonymousGraphCardMultisetRequest(StrictModel):
    """Unordered finite multiset input; card_order disambiguates the empty case."""

    card_order: int = Field(ge=0, le=MAX_UNLABELLED_DECK_VERTICES)
    cards: tuple[SimpleUndirectedGraph, ...] = Field(
        max_length=MAX_ANONYMOUS_CARD_CLASSES,
        description=(
            "An unordered list of simple graphs, each with exactly card_order "
            "vertices. Repeated isomorphic cards encode multiplicity. An empty "
            "list is the empty multiset of cards of the declared order."
        ),
    )

    @model_validator(mode="after")
    def require_declared_order(self) -> Self:
        if any(len(card.vertices) != self.card_order for card in self.cards):
            raise _validation_error(
                "anonymous_card_order", "every card must have the declared card_order"
            )
        return self


class AnonymousGraphCardClass(StrictModel):
    """One canonically relabelled isomorphism class and its exact multiplicity."""

    representative: SimpleUndirectedGraph
    multiplicity: Annotated[int, DecimalIntegerEncoding(max_digits=12)] = Field(ge=1)


class AnonymousGraphCardMultiset(StrictModel):
    """Anonymous card multiset; it carries no source graph or deletion keys."""

    card_order: int = Field(ge=0, le=MAX_UNLABELLED_DECK_VERTICES)
    classes: tuple[AnonymousGraphCardClass, ...] = Field(
        max_length=MAX_ANONYMOUS_CARD_CLASSES
    )

    @model_validator(mode="after")
    def require_structural_canonical_form(self) -> Self:
        n = self.card_order
        if type(n) is not int or n < 0 or n > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "anonymous_card_order", "card_order is outside its bound"
            )
        if (
            type(self.classes) is not tuple
            or len(self.classes) > MAX_ANONYMOUS_CARD_CLASSES
        ):
            raise _validation_error(
                "anonymous_card_classes", "classes exceed the carrier bound"
            )
        pair_count = comb(n, 2)
        work = _anonymous_canonicalization_work(n, len(self.classes))
        if work > MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK:
            raise _validation_error(
                "anonymous_card_validation_bound",
                "canonical class validation exceeds its bounded permutation work",
            )
        output_bytes = len(self.classes) * (64 + 16 * pair_count)
        if output_bytes > MAX_ANONYMOUS_CARD_RESULT_BYTES:
            raise _validation_error(
                "anonymous_card_output_bound",
                "canonical classes exceed the result byte bound",
            )
        previous_key: tuple[tuple[str, str], ...] | None = None
        expected_vertices = tuple(f"v{i:02d}" for i in range(n))
        for item in self.classes:
            if type(item) is not AnonymousGraphCardClass:
                raise _validation_error(
                    "anonymous_card_class_carrier", "classes have the wrong carrier"
                )
            multiplicity = getattr(item, "multiplicity", None)
            if (
                type(multiplicity) is not int
                or multiplicity < 1
                or multiplicity >= 10**12
            ):
                raise _validation_error(
                    "anonymous_card_multiplicity",
                    "class multiplicity must be a positive bounded exact integer",
                )
            graph = getattr(item, "representative", None)
            if (
                type(graph) is not SimpleUndirectedGraph
                or type(graph.vertices) is not tuple
                or len(graph.vertices) != n
                or graph.vertices != expected_vertices
                or type(graph.edges) is not tuple
                or len(graph.edges) > pair_count
            ):
                raise _validation_error(
                    "anonymous_card_labels",
                    "representatives must have the canonical fixed-width axis and bounded edges",
                )
            if any(
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(label) is not str for label in edge)
                for edge in graph.edges
            ):
                raise _validation_error(
                    "anonymous_card_edges", "representative edges must be pairs"
                )
            if (
                len(set(graph.edges)) != len(graph.edges)
                or any(
                    left >= right
                    or left not in expected_vertices
                    or right not in expected_vertices
                    for left, right in graph.edges
                )
                or tuple(sorted(graph.edges)) != graph.edges
            ):
                raise _validation_error(
                    "anonymous_card_edges",
                    "representative edges must be valid, unique, and ordered",
                )
            if _canonical_card_edges(graph.vertices, graph.edges) != graph.edges:
                raise _validation_error(
                    "anonymous_card_not_canonical",
                    "each representative must be minimal under all vertex permutations",
                )
            if previous_key is not None and graph.edges <= previous_key:
                raise _validation_error(
                    "anonymous_card_classes",
                    "classes must be unique and lexicographically ordered",
                )
            previous_key = graph.edges
        return self

    @classmethod
    def _from_kernel(
        cls, card_order: int, classes: tuple[AnonymousGraphCardClass, ...]
    ) -> Self:
        return cls.model_construct(card_order=card_order, classes=classes)


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
    canonical shape. Kernel output uses ``_from_kernel`` after bounded
    construction; the appearance ledgers follow directly from deletion.
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
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class UnlabelledEdgeDeckRequest(StrictModel):
    """Consume a complete source-bound edge-deletion family."""

    deck: EdgeDeletionFamily = Field(
        description=(
            "A complete source-bound edge deck; exact permutation canonicalization "
            "is admitted by source order, aggregate work, and result bytes."
        )
    )


class UnlabelledEdgeDeckClass(StrictModel):
    """One graph-isomorphism class with exact source-edge provenance."""

    representative: SimpleUndirectedGraph
    multiplicity: int = Field(ge=1)
    card_indices: tuple[int, ...]
    deleted_edges: tuple[tuple[str, str], ...]


class UnlabelledEdgeDeck(StrictModel):
    """Exact multiset quotient of a source-bound edge-deletion family."""

    family: EdgeDeletionFamily
    classes: tuple[UnlabelledEdgeDeckClass, ...]
    card_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_partition(self) -> Self:
        order = len(self.family.source.vertices)
        work = len(self.family.cards) * factorial(order) * (1 + order + comb(order, 2))
        if (
            order > MAX_UNLABELLED_DECK_VERTICES
            or work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK
        ):
            raise _validation_error(
                "edge_quotient_bound",
                "unlabelled edge deck exceeds its exact canonicalization envelope",
            )
        if self.card_count != len(self.family.cards) or self.card_count != len(
            self.family.source.edges
        ):
            raise _validation_error(
                "edge_quotient_card_count", "deck must retain every source edge card"
            )
        seen: list[int] = []
        for item in self.classes:
            if (
                item.multiplicity != len(item.card_indices)
                or item.multiplicity != len(item.deleted_edges)
                or not item.card_indices
                or tuple(sorted(item.card_indices)) != item.card_indices
                or any(
                    index < 0 or index >= self.card_count for index in item.card_indices
                )
                or item.deleted_edges
                != tuple(
                    self.family.cards[index].deleted_edge for index in item.card_indices
                )
                or item.representative != self.family.cards[item.card_indices[0]].card
            ):
                raise _validation_error(
                    "edge_quotient_class_provenance",
                    "each class must retain aligned card indices, source edges, and its first source card",
                )
            seen.extend(item.card_indices)
        if sorted(seen) != list(range(self.card_count)):
            raise _validation_error(
                "edge_quotient_indices",
                "class indices must partition the edge-card axis",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class UnlabelledVertexDeckRequest(StrictModel):
    """Consume a complete source-bound vertex-deletion family."""

    deck: VertexDeletionFamily = Field(
        description=(
            "A complete vertex-deletion family with at most 10 source vertices; "
            "exact permutation canonicalization is bounded by 2000000 work units."
        )
    )


class UnlabelledVertexDeckClass(StrictModel):
    """One isomorphism class in a vertex-deck multiset."""

    representative: SimpleUndirectedGraph
    multiplicity: int = Field(ge=1)
    card_indices: tuple[int, ...]


class UnlabelledVertexDeck(StrictModel):
    """Exact multiset quotient of a source-bound vertex-deletion family."""

    family: VertexDeletionFamily
    classes: tuple[UnlabelledVertexDeckClass, ...]
    card_count: int = Field(ge=0)

    @model_validator(mode="after")
    def require_partition(self) -> Self:
        family = self.family
        if len(family.source.vertices) > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "vertex_quotient_bound",
                "unlabelled deck exceeds the isomorphism envelope",
            )
        card_order = max(len(family.source.vertices) - 1, 0)
        if (
            len(family.cards)
            * factorial(card_order)
            * (1 + card_order + comb(card_order, 2))
            > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK
        ):
            raise _validation_error(
                "vertex_quotient_work_bound",
                "unlabelled deck exceeds the exact permutation work envelope",
            )
        if (
            self.card_count != len(family.cards)
            or sum(item.multiplicity for item in self.classes) != self.card_count
        ):
            raise _validation_error(
                "vertex_quotient_card_count", "classes must partition every vertex card"
            )
        indices = [index for item in self.classes for index in item.card_indices]
        if sorted(indices) != list(range(self.card_count)):
            raise _validation_error(
                "vertex_quotient_indices", "class indices must partition the card axis"
            )
        for item in self.classes:
            if (
                len(item.card_indices) != item.multiplicity
                or tuple(sorted(item.card_indices)) != item.card_indices
                or not item.card_indices
                or item.card_indices[-1] >= len(family.cards)
                or item.representative != family.cards[item.card_indices[0]].card
            ):
                raise _validation_error(
                    "vertex_quotient_representative",
                    "class representative must be its first indexed source card",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class VertexDeckInducedSubgraphCountRequest(StrictModel):
    """Count an induced pattern from a complete vertex deck."""

    deck: UnlabelledVertexDeck
    pattern: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_proper_pattern(self) -> Self:
        if len(self.pattern.vertices) >= len(self.deck.family.source.vertices):
            raise _validation_error(
                "kelly_pattern_not_proper",
                "the induced pattern order must be strictly smaller than deck source order",
            )
        return self


class VertexDeckInducedSubgraphContribution(StrictModel):
    """One isomorphism-class contribution to Kelly's double count."""

    class_index: int = Field(ge=0)
    representative_card: SimpleUndirectedGraph
    card_indices: tuple[int, ...]
    multiplicity: int = Field(ge=1)
    occurrences_per_card: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_COUNT_DIGITS)
    ] = Field(ge=0)
    weighted_occurrences: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_COUNT_DIGITS)
    ] = Field(ge=0)


class VertexDeckInducedSubgraphCount(StrictModel):
    """Induced subset count reconstructed from a complete vertex deck."""

    deck: UnlabelledVertexDeck
    pattern: SimpleUndirectedGraph
    contributions: tuple[VertexDeckInducedSubgraphContribution, ...]
    weighted_card_total: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_COUNT_DIGITS)
    ] = Field(ge=0)
    overcount_divisor: int = Field(ge=1)
    occurrence_count: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_COUNT_DIGITS)
    ] = Field(ge=0)

    @model_validator(mode="after")
    def require_kelly_identity(self) -> Self:
        source_order = len(self.deck.family.source.vertices)
        pattern_order = len(self.pattern.vertices)
        if pattern_order >= source_order:
            raise _validation_error(
                "kelly_pattern_not_proper",
                "the induced pattern order must be strictly smaller than deck source order",
            )
        if self.overcount_divisor != source_order - pattern_order:
            raise _validation_error(
                "kelly_divisor",
                "the overcount divisor must equal n minus pattern order",
            )
        classes = self.deck.classes
        if len(self.contributions) != len(classes):
            raise _validation_error(
                "kelly_contribution_count",
                "there must be one contribution for every deck isomorphism class",
            )
        total = 0
        for index, (row, card_class) in enumerate(
            zip(self.contributions, classes, strict=True)
        ):
            if (
                row.class_index != index
                or row.representative_card != card_class.representative
                or row.card_indices != card_class.card_indices
                or row.multiplicity != card_class.multiplicity
                or row.weighted_occurrences
                != row.occurrences_per_card * row.multiplicity
            ):
                raise _validation_error(
                    "kelly_contribution_binding",
                    "each contribution must bind its deck class and multiplicity",
                )
            total += row.weighted_occurrences
        if total != self.weighted_card_total:
            raise _validation_error(
                "kelly_weighted_total", "weighted contributions must sum to the total"
            )
        if total % self.overcount_divisor:
            raise _validation_error(
                "kelly_nondivisible",
                "weighted card total must be divisible by n minus pattern order",
            )
        if total // self.overcount_divisor != self.occurrence_count:
            raise _validation_error(
                "kelly_occurrence_count",
                "occurrence count must equal the Kelly quotient",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class VertexDeckSubgraphCountRequest(StrictModel):
    """Count ordinary (not necessarily induced) copies of a proper pattern."""

    deck: UnlabelledVertexDeck
    pattern: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_proper_pattern(self) -> Self:
        if len(self.pattern.vertices) >= len(self.deck.family.source.vertices):
            raise _validation_error(
                "kelly_subgraph_pattern_not_proper",
                "the subgraph pattern order must be strictly smaller than deck source order",
            )
        return self


class VertexDeckSubgraphContribution(StrictModel):
    """One deck isomorphism-class contribution to the ordinary subgraph count."""

    class_index: int = Field(ge=0)
    representative_card: SimpleUndirectedGraph
    card_indices: tuple[int, ...]
    multiplicity: int = Field(ge=1)
    occurrences_per_card: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_SUBGRAPH_COUNT_DIGITS)
    ] = Field(ge=0)
    weighted_occurrences: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_SUBGRAPH_COUNT_DIGITS)
    ] = Field(ge=0)


class VertexDeckSubgraphCount(StrictModel):
    """Source-bound count of edge-subset copies of a proper graph pattern.

    A copy is a pair of a vertex subset and an edge subset on it isomorphic to
    ``pattern``; additional host edges on the same vertices are allowed. This
    differs from both induced copies and injective embeddings.
    """

    deck: UnlabelledVertexDeck
    pattern: SimpleUndirectedGraph
    contributions: tuple[VertexDeckSubgraphContribution, ...]
    weighted_card_total: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_SUBGRAPH_COUNT_DIGITS)
    ] = Field(ge=0)
    overcount_divisor: int = Field(ge=1)
    occurrence_count: Annotated[
        int, DecimalIntegerEncoding(max_digits=MAX_KELLY_SUBGRAPH_COUNT_DIGITS)
    ] = Field(ge=0)

    @model_validator(mode="after")
    def require_kelly_identity(self) -> Self:
        source_order = len(self.deck.family.source.vertices)
        pattern_order = len(self.pattern.vertices)
        if pattern_order >= source_order:
            raise _validation_error(
                "kelly_subgraph_pattern_not_proper",
                "the subgraph pattern order must be strictly smaller than deck source order",
            )
        if self.overcount_divisor != source_order - pattern_order:
            raise _validation_error(
                "kelly_subgraph_divisor",
                "the overcount divisor must equal n minus pattern order",
            )
        if len(self.contributions) != len(self.deck.classes):
            raise _validation_error(
                "kelly_subgraph_contribution_count",
                "there must be one contribution for every deck isomorphism class",
            )
        total = 0
        for index, (row, card_class) in enumerate(
            zip(self.contributions, self.deck.classes, strict=True)
        ):
            if (
                row.class_index != index
                or row.representative_card != card_class.representative
                or row.card_indices != card_class.card_indices
                or row.multiplicity != card_class.multiplicity
                or row.weighted_occurrences
                != row.occurrences_per_card * row.multiplicity
            ):
                raise _validation_error(
                    "kelly_subgraph_contribution_binding",
                    "each contribution must bind its deck class and multiplicity",
                )
            total += row.weighted_occurrences
        if total != self.weighted_card_total:
            raise _validation_error(
                "kelly_subgraph_weighted_total",
                "weighted contributions must sum to the total",
            )
        if (
            total % self.overcount_divisor
            or total // self.overcount_divisor != self.occurrence_count
        ):
            raise _validation_error(
                "kelly_subgraph_occurrence_count",
                "occurrence count must equal the Kelly quotient",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class VertexDeckEdgeCountRequest(StrictModel):
    """Reconstruct source edge count from an exact unlabelled vertex deck."""

    deck: UnlabelledVertexDeck


class VertexDeckDegreeMultisetRequest(StrictModel):
    """Recover the source degree multiset from a complete vertex deck."""

    deck: UnlabelledVertexDeck


class VertexDeckEdgeCount(StrictModel):
    """Exact source edge count reconstructed by the vertex-deck identity."""

    deck: UnlabelledVertexDeck
    card_edge_counts: tuple[int, ...] = Field(
        min_length=3, max_length=MAX_UNLABELLED_DECK_VERTICES
    )
    card_edge_total: int = Field(ge=0, le=MAX_VERTEX_DECK_CARD_EDGE_TOTAL)
    overcount_divisor: int = Field(ge=1, le=MAX_UNLABELLED_DECK_VERTICES - 2)
    source_edge_count: int = Field(ge=0, le=MAX_VERTEX_DECK_SOURCE_EDGES)

    @model_validator(mode="after")
    def require_edge_count_identity(self) -> Self:
        source_order = self.deck.card_count
        expected = tuple(
            sorted(
                len(card_class.representative.edges)
                for card_class in self.deck.classes
                for _ in range(card_class.multiplicity)
            )
        )
        if source_order < 3:
            raise _validation_error(
                "edge_count_order",
                "vertex-deck edge count requires at least three cards",
            )
        if self.card_edge_counts != expected:
            raise _validation_error(
                "edge_count_card_profile",
                "card edge counts must be the exact multiset from deck classes",
            )
        if self.overcount_divisor != source_order - 2:
            raise _validation_error(
                "edge_count_divisor", "the divisor must equal source order minus two"
            )
        total = sum(self.card_edge_counts)
        if total != self.card_edge_total:
            raise _validation_error(
                "edge_count_total",
                "card edge total must sum the complete card multiset",
            )
        if total % self.overcount_divisor:
            raise _validation_error(
                "edge_count_nondivisible",
                "card edge total must be divisible by source order minus two",
            )
        if total // self.overcount_divisor != self.source_edge_count:
            raise _validation_error(
                "edge_count_result",
                "source edge count must equal the exact deck quotient",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_DECK_CARD_EDGES",
    "MAX_DECK_VERTICES",
    "MAX_EDGE_DECK_EDGES",
    "MAX_KELLY_COUNT_DIGITS",
    "MAX_KELLY_DECK_TOTAL_WORK",
    "MAX_KELLY_RESULT_BYTES",
    "MAX_KELLY_SUBGRAPH_COUNT_DIGITS",
    "MAX_UNLABELLED_DECK_ISOMORPHISM_WORK",
    "MAX_UNLABELLED_DECK_VERTICES",
    "MAX_VERTEX_DECK_CARD_EDGE_TOTAL",
    "MAX_VERTEX_DECK_SOURCE_EDGES",
    "EdgeDeckRequest",
    "EdgeDeletionFamily",
    "SourceBoundEdgeCard",
    "SourceBoundVertexCard",
    "UnlabelledDeck",
    "UnlabelledDeckClass",
    "UnlabelledDeckRequest",
    "UnlabelledVertexDeck",
    "UnlabelledVertexDeckClass",
    "UnlabelledVertexDeckRequest",
    "VertexDeckDegreeMultisetRequest",
    "VertexDeckEdgeCount",
    "VertexDeckEdgeCountRequest",
    "VertexDeckInducedSubgraphContribution",
    "VertexDeckInducedSubgraphCount",
    "VertexDeckInducedSubgraphCountRequest",
    "VertexDeckRequest",
    "VertexDeckSubgraphContribution",
    "VertexDeckSubgraphCount",
    "VertexDeckSubgraphCountRequest",
    "VertexDeletionFamily",
]
