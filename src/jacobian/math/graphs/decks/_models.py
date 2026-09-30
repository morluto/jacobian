"""Typed wire contracts for exact vertex-deletion deck operations."""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb, factorial
from typing import Annotated, Any, Self, cast

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import DecimalIntegerEncoding
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.math.graphs.patterns._models import (
    MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS,
)
from jacobian.math.graphs.values import MAX_GRAPH_LABEL_BYTES, SimpleUndirectedGraph

MAX_DECK_VERTICES = 64
"""Admission cap on source vertices so the complete card family fits output."""

MAX_DECK_CARD_EDGES = 130_000
"""Admission cap on aggregate card edges across the whole family."""
MAX_EDGE_DECK_EDGES = 130_000
MAX_UNLABELLED_DECK_VERTICES = 10
MAX_UNLABELLED_DECK_ISOMORPHISM_WORK = 2_000_000
MAX_UNLABELLED_EDGE_RESULT_UNITS = 1_000_000
MAX_EDGE_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS = 1_000_000
MAX_VERTEX_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS = 1_000_000
MAX_ANONYMOUS_CARD_CANONICALIZATION_WORK = 2_000_000
MAX_ANONYMOUS_CARD_RESULT_UNITS = 1_000_000
MAX_ANONYMOUS_CARD_CLASSES = MAX_ANONYMOUS_CARD_RESULT_UNITS // 64
"""Admission cap on aggregate card edges across the whole family."""
MAX_VERTEX_DECK_SOURCE_EDGES = comb(MAX_UNLABELLED_DECK_VERTICES, 2)
MAX_VERTEX_DECK_CARD_EDGE_TOTAL = MAX_UNLABELLED_DECK_VERTICES * comb(
    MAX_UNLABELLED_DECK_VERTICES - 1, 2
)

MAX_KELLY_DECK_TOTAL_WORK = MAX_INDUCED_PATTERN_TOTAL_WORK_UNITS
MAX_KELLY_COUNT_DIGITS = 3
MAX_KELLY_SUBGRAPH_COUNT_DIGITS = 12
MAX_DECK_ECHO_ALLOCATION = 1_000_000
"""Bound on vertex-label characters a Kelly deck result may echo.

A deck result echoes its source family, one class representative per deck
class, and one contribution representative per class. Each label appears at
most once per incident graph record, so bounding the aggregate label
allocation before canonicalization bounds the exact output growth.
"""

MAX_DEGREE_MULTISET_DIGITS = 64
"""Aggregate decimal digits the reconstructed degree multiset may allocate."""


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
    if best is None:
        raise ValueError("permutation orbit must contain at least one ordering")
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
    multiplicity: Annotated[int, DecimalIntegerEncoding(max_digits=12)] = Field(
        ge=1,
        json_schema_extra={
            "pattern": "^[1-9][0-9]{0,11}(?![\\s\\S])",
            "maxLength": 12,
            "maximum": 999_999_999_999,
        },
    )


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
        output_units = len(self.classes) * (64 + 16 * pair_count)
        if output_units > MAX_ANONYMOUS_CARD_RESULT_UNITS:
            raise _validation_error(
                "anonymous_card_output_bound",
                "canonical classes exceed the admitted result allocation bound",
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


class VertexDeckAnonymousMultisetRequest(StrictModel):
    """Forget source identities from one complete vertex-deletion family."""

    family: VertexDeletionFamily = Field(
        description=(
            "A complete exact source-bound vertex-deletion family, including "
            "all cards and aligned receipts. The operation forgets source labels "
            "after authenticating the family and returns its anonymous graph-card "
            "multiset; source order is limited by the exact permutation work "
            "envelope, including self-comparison closure: 2*n*(n-1)!* "
            "((n-1) + 2*binom(n-1, 2)) <= 2,000,000 (n <= 7)."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_family(cls, value: Any) -> Any:
        """Apply the cheap source-order cap before nested card parsing."""
        if not isinstance(value, dict):
            return value
        raw_family = value.get("family")
        if not isinstance(raw_family, dict):
            return value
        raw_source = raw_family.get("source")
        if not isinstance(raw_source, dict):
            return value
        vertices = raw_source.get("vertices")
        if not isinstance(vertices, (tuple, list)):
            return value
        source_order = len(vertices)
        if source_order > MAX_UNLABELLED_DECK_VERTICES + 1:
            raise _validation_error(
                "anonymous_source_order",
                "source order exceeds the cheap structural envelope of 11 vertices",
            )
        return canonicalize_json_containers(value)


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
            "10 source vertices and 2000000 units of exact permutation "
            "canonicalization work."
        ),
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


class UnlabelledVertexDeckRequest(StrictModel):
    """Consume a complete source-bound vertex-deletion family."""

    deck: VertexDeletionFamily = Field(
        description=(
            "A complete vertex-deletion family admitted through source order 8 "
            "under n*(n-1)!*(1+(n-1)+binom(n-1, 2)) <= 2000000 exact "
            "permutation-canonicalization work units."
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
    "MAX_DECK_ECHO_ALLOCATION",
    "MAX_DECK_VERTICES",
    "MAX_DEGREE_MULTISET_DIGITS",
    "MAX_EDGE_DECK_EDGES",
    "MAX_KELLY_COUNT_DIGITS",
    "MAX_KELLY_DECK_TOTAL_WORK",
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


class VertexDeckIsomorphismProfileRequest(StrictModel):
    """Produce exact card-to-class maps for a complete vertex-deletion family."""

    deck: VertexDeletionFamily

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source_order(cls, value: Any) -> Any:
        if type(value) is not dict:
            return value
        deck = value.get("deck")
        if type(deck) is not dict:
            return value
        dimensions = _vertex_iso_profile_dimensions(deck, None)
        if dimensions is None:
            raise _validation_error(
                "vertex_iso_profile_family",
                "profile family must retain its source vertex axis",
            )
        order, source_edges, _ = dimensions
        if order > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "vertex_iso_profile_bound",
                "vertex-deck isomorphism profile supports at most 10 source vertices",
            )
        _preflight_vertex_profile_family(deck, order, source_edges)
        _, total_work, output_cells = _vertex_iso_profile_resource_estimates(
            order, source_edges, order
        )
        if total_work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
            raise _validation_error(
                "vertex_iso_profile_work_bound",
                "vertex-deck isomorphism mapping exceeds the shared work bound",
            )
        if output_cells > MAX_VERTEX_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS:
            raise _validation_error(
                "vertex_iso_profile_output_bound",
                "vertex-deck isomorphism profile exceeds its materialization-cell bound",
            )
        return _normalize_vertex_iso_profile_request(value)


class VertexDeckIsomorphismClass(StrictModel):
    """One canonical graph class and its source-card indices."""

    representative: SimpleUndirectedGraph
    multiplicity: int = Field(ge=1)
    card_indices: tuple[int, ...]


class VertexDeckIsomorphismProfile(StrictModel):
    """Source-bound card partition and explicit vertex bijections.

    The producer establishes canonical representatives and the exact
    isomorphism quotient. Decoding checks bounded structure, the partition,
    and the supplied bijections; it does not authenticate an authored claim
    that differently labelled representatives are nonisomorphic. To establish
    that claim, run ``vertex_deck_isomorphism_profile`` on the retained family
    under its native admission rather than trusting a decoded class count.
    """

    family: VertexDeletionFamily
    classes: tuple[VertexDeckIsomorphismClass, ...] = Field(
        description=(
            "Canonical isomorphism classes established by the producer. "
            "Decoding validates their structural partition and maps without "
            "repeating the canonicalization search or proving class distinctness."
        )
    )
    class_indices: tuple[int, ...]
    vertex_maps: tuple[tuple[int, ...], ...]

    @model_validator(mode="before")
    @classmethod
    def normalize_json_tuple_fields(cls, value: Any) -> Any:
        return _admit_and_normalize_vertex_iso_profile_result(value)

    @model_validator(mode="after")
    def require_exact_partition_and_maps(self) -> Self:
        family = self.family
        if (
            type(family) is not VertexDeletionFamily
            or type(family.source) is not SimpleUndirectedGraph
            or type(family.source.vertices) is not tuple
            or type(family.source.edges) is not tuple
            or type(family.cards) is not tuple
        ):
            raise _validation_error(
                "vertex_iso_profile_family",
                "profile must retain a canonical source-bound vertex family",
            )
        source_order = len(family.source.vertices)
        card_order = max(source_order - 1, 0)
        card_count = len(family.cards)
        if source_order > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "vertex_iso_profile_bound",
                "vertex-deck isomorphism profile exceeds its source-order bound",
            )
        if (
            len(self.class_indices) != card_count
            or len(self.vertex_maps) != card_count
            or len(self.classes) > card_count
            or sum(item.multiplicity for item in self.classes) != card_count
        ):
            raise _validation_error(
                "vertex_iso_profile_card_count",
                "profile rows and card maps must cover every source card",
            )

        previous_edges: tuple[tuple[str, str], ...] | None = None
        seen: list[int] = []
        canonical_axis = tuple(f"v{i:02d}" for i in range(card_order))
        for class_index, item in enumerate(self.classes):
            representative = item.representative
            if (
                type(item.card_indices) is not tuple
                or not item.card_indices
                or tuple(sorted(item.card_indices)) != item.card_indices
                or item.multiplicity != len(item.card_indices)
                or any(
                    type(index) is not int or index < 0 or index >= card_count
                    for index in item.card_indices
                )
                or representative.vertices != canonical_axis
                # See the edge-profile validator: the canonical adjacency form
                # is the producer's postcondition, not something result
                # decoding re-derives on every round trip.
                or (
                    previous_edges is not None
                    and representative.edges <= previous_edges
                )
            ):
                raise _validation_error(
                    "vertex_iso_profile_class",
                    "classes must be unique, canonical, ordered, and cover valid card indices",
                )
            previous_edges = representative.edges
            seen.extend(item.card_indices)
            for card_index in item.card_indices:
                if self.class_indices[card_index] != class_index:
                    raise _validation_error(
                        "vertex_iso_profile_class_map",
                        "class_indices must agree with each class card list",
                    )
        if sorted(seen) != list(range(card_count)):
            raise _validation_error(
                "vertex_iso_profile_partition",
                "class card indices must partition the card axis",
            )

        for card_index, (class_index, mapping) in enumerate(
            zip(self.class_indices, self.vertex_maps, strict=True)
        ):
            if (
                type(class_index) is not int
                or class_index < 0
                or class_index >= len(self.classes)
                or type(mapping) is not tuple
                or len(mapping) != card_order
                or any(type(value) is not int for value in mapping)
                or tuple(sorted(mapping)) != tuple(range(card_order))
            ):
                raise _validation_error(
                    "vertex_iso_profile_map_shape",
                    "each card needs one valid class index and vertex permutation",
                )
            card = family.cards[card_index].card
            edge_index = {
                vertex: position for position, vertex in enumerate(card.vertices)
            }
            target_edges = tuple(
                sorted(
                    (
                        (
                            f"v{min(mapping[edge_index[left]], mapping[edge_index[right]]):02d}",
                            f"v{max(mapping[edge_index[left]], mapping[edge_index[right]]):02d}",
                        )
                        for left, right in card.edges
                    )
                )
            )
            if target_edges != self.classes[class_index].representative.edges:
                raise _validation_error(
                    "vertex_iso_profile_map_relation",
                    "each vertex map must carry its card edges to the class representative",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _vertex_iso_profile_output_bound(
    family: VertexDeletionFamily, class_count: int
) -> int:
    return _vertex_iso_profile_resource_estimates(
        len(family.source.vertices), len(family.source.edges), class_count
    )[2]


def _normalize_vertex_iso_profile_request(value: Any) -> Any:
    if type(value) is not dict:
        return value
    deck = value.get("deck")
    if type(deck) is not dict:
        return value
    normalized = dict(value)
    normalized["deck"] = _normalize_vertex_family_json(deck)
    return normalized


def _normalize_vertex_iso_profile_result(value: Any) -> Any:
    if type(value) is not dict:
        return value
    family = value.get("family")
    classes = value.get("classes")
    normalized = dict(value)
    if type(family) is dict:
        normalized["family"] = _normalize_vertex_family_json(family)
    if type(classes) is list:
        normalized_classes = []
        for item in classes:
            if type(item) is not dict:
                normalized_classes.append(item)
                continue
            row = dict(item)
            indices = row.get("card_indices")
            if type(indices) is list:
                row["card_indices"] = tuple(indices)
            representative = row.get("representative")
            if type(representative) is dict:
                row["representative"] = _normalize_vertex_graph_json(representative)
            normalized_classes.append(row)
        normalized["classes"] = tuple(normalized_classes)
    for field in ("class_indices", "vertex_maps"):
        rows = normalized.get(field)
        if type(rows) is list:
            normalized[field] = tuple(
                tuple(row) if field == "vertex_maps" and type(row) is list else row
                for row in rows
            )
    return normalized


def _admit_and_normalize_vertex_iso_profile_result(value: Any) -> Any:
    if type(value) is not dict:
        return value
    dimensions = _vertex_iso_profile_dimensions(
        value.get("family"), value.get("classes")
    )
    if dimensions is None:
        raise _validation_error(
            "vertex_iso_profile_family",
            "profile family must retain its source vertex axis",
        )
    order, edge_count, class_count = dimensions
    if order > MAX_UNLABELLED_DECK_VERTICES:
        raise _validation_error(
            "vertex_iso_profile_bound",
            "vertex-deck isomorphism profile exceeds its source-order bound",
        )
    classes = value.get("classes")
    if (
        type(classes) in (list, tuple)
        and len(cast(list[Any] | tuple[Any, ...], classes)) > order
    ):
        raise _validation_error(
            "vertex_iso_profile_class_count",
            "the isomorphism profile cannot have more classes than cards",
        )
    _, work, output_cells = _vertex_iso_profile_value_resource_estimates(
        order, edge_count, class_count
    )
    if work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise _validation_error(
            "vertex_iso_profile_validation_work_bound",
            "class representatives and card maps exceed the shared validation work bound",
        )
    if output_cells > MAX_VERTEX_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS:
        raise _validation_error(
            "vertex_iso_profile_output_bound",
            "vertex-deck isomorphism profile exceeds its materialization-cell bound",
        )
    _preflight_vertex_profile_result_rows(value, order)
    _preflight_vertex_profile_family(value.get("family"), order, edge_count)
    _require_exact_profile_wire_integers(value, classes, "vertex")
    return _normalize_vertex_iso_profile_result(value)


def _preflight_vertex_profile_result_rows(value: dict[str, Any], order: int) -> None:
    """Bound the vertex-profile rows before the normalizer copies them.

    The normalizer materializes every attacker-controlled list as tuples, so an
    oversized ``vertex_maps`` row or class ``card_indices`` list would be fully
    copied before the after-validator could reject its shape. The edge-profile
    decoder already preflights the same axes.
    """
    card_count = order
    card_order = max(order - 1, 0)
    class_indices: Any = value.get("class_indices")
    if type(class_indices) in (list, tuple) and len(class_indices) > card_count:
        raise _validation_error(
            "vertex_iso_profile_card_count",
            "class index rows exceed the source-card axis",
        )
    maps: Any = value.get("vertex_maps")
    if type(maps) in (list, tuple):
        if len(maps) > card_count:
            raise _validation_error(
                "vertex_iso_profile_card_count",
                "vertex map rows exceed the source-card axis",
            )
        if any(
            type(row) in (list, tuple) and len(row) > card_order
            for row in cast(list[Any] | tuple[Any, ...], maps)
        ):
            raise _validation_error(
                "vertex_iso_profile_map_shape",
                "a vertex map exceeds the source vertex axis",
            )
    classes: Any = value.get("classes")
    if type(classes) not in (list, tuple):
        return
    for item in cast(list[Any] | tuple[Any, ...], classes):
        if type(item) is not dict:
            continue
        indices: Any = item.get("card_indices")
        if type(indices) in (list, tuple) and len(indices) > card_count:
            raise _validation_error(
                "vertex_iso_profile_class_shape",
                "class card indices exceed the source-card axis",
            )
        representative: Any = item.get("representative")
        if type(representative) is not dict:
            continue
        representative_vertices: Any = representative.get("vertices")
        representative_edges: Any = representative.get("edges")
        if (
            type(representative_vertices) in (list, tuple)
            and len(representative_vertices) > card_order
        ) or (
            type(representative_edges) in (list, tuple)
            and len(representative_edges) > comb(card_order, 2)
        ):
            raise _validation_error(
                "vertex_iso_profile_class_shape",
                "a class representative exceeds the card-order shape bound",
            )
        _preflight_raw_edge_pairs(representative_edges, comb(card_order, 2))
        _preflight_edge_profile_labels(representative_vertices, representative_edges)


def _preflight_vertex_profile_family(family: Any, order: int, edge_count: int) -> None:
    """Bound all retained source axes before the family normalizer copies them."""
    if type(family) is not dict:
        return
    pair_count = comb(order, 2)
    card_order = max(order - 1, 0)
    card_pair_count = comb(card_order, 2)
    if edge_count > pair_count:
        raise _validation_error(
            "vertex_iso_profile_source_edges",
            "source edge list exceeds the simple-graph order bound",
        )
    source = family.get("source")
    if type(source) is dict:
        _preflight_raw_edge_pairs(source.get("edges"), pair_count)
    cards: Any = family.get("cards")
    if type(cards) in (list, tuple):
        if len(cards) != order:
            raise _validation_error(
                "vertex_iso_profile_card_count",
                "a complete vertex family must contain one card per source vertex",
            )
        for card in cards:
            if type(card) is not dict:
                continue
            retained: Any = card.get("retained_vertices")
            if type(retained) in (list, tuple) and len(retained) > card_order:
                raise _validation_error(
                    "vertex_iso_profile_card_shape",
                    "retained vertices exceed the declared card-order bound",
                )
            graph = card.get("card")
            if type(graph) is not dict:
                continue
            for field, maximum in (
                ("vertices", card_order),
                ("edges", card_pair_count),
            ):
                entries: Any = graph.get(field)
                if type(entries) in (list, tuple) and len(entries) > maximum:
                    raise _validation_error(
                        "vertex_iso_profile_card_shape",
                        "a card exceeds the declared source-order shape bound",
                    )
            _preflight_raw_edge_pairs(graph.get("edges"), card_pair_count)
    for field, maximum in (
        ("edge_appearances", edge_count),
        ("vertex_appearances", order),
    ):
        ledger: Any = family.get(field)
        if type(ledger) in (list, tuple) and len(ledger) > maximum:
            raise _validation_error(
                "vertex_iso_profile_ledger_shape",
                "appearance ledgers exceed their retained source axes",
            )


def _vertex_iso_profile_dimensions(
    family: Any, classes: Any
) -> tuple[int, int, int] | None:
    if type(family) is VertexDeletionFamily:
        source = getattr(family, "source", None)
        vertices = (
            getattr(source, "vertices", None)
            if type(source) is SimpleUndirectedGraph
            else None
        )
        edges = (
            getattr(source, "edges", None)
            if type(source) is SimpleUndirectedGraph
            else None
        )
    elif type(family) is dict:
        raw_source: Any = family.get("source")
        if type(raw_source) is SimpleUndirectedGraph:
            vertices = getattr(raw_source, "vertices", None)
            edges = getattr(raw_source, "edges", None)
        elif type(raw_source) is dict:
            vertices = raw_source.get("vertices")
            edges = raw_source.get("edges")
        else:
            return None
    else:
        return None
    if type(vertices) not in (list, tuple):
        return None
    order = len(cast(list[Any] | tuple[Any, ...], vertices))
    edge_count = (
        len(cast(list[Any] | tuple[Any, ...], edges))
        if type(edges) in (list, tuple)
        else comb(order, 2)
    )
    class_count = len(classes) if type(classes) in (list, tuple) else order
    return order, edge_count, class_count


def _normalize_vertex_family_json(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    source = normalized.get("source")
    if type(source) is dict:
        normalized["source"] = _normalize_vertex_graph_json(source)
    cards = normalized.get("cards")
    if type(cards) is list:
        normalized_cards = []
        for item in cards:
            if type(item) is not dict:
                normalized_cards.append(item)
                continue
            row = dict(item)
            card = row.get("card")
            if type(card) is dict:
                row["card"] = _normalize_vertex_graph_json(card)
            retained = row.get("retained_vertices")
            if type(retained) is list:
                row["retained_vertices"] = tuple(retained)
            normalized_cards.append(row)
        normalized["cards"] = tuple(normalized_cards)
    for field in ("edge_appearances", "vertex_appearances"):
        entries = normalized.get(field)
        if type(entries) is list:
            normalized[field] = tuple(entries)
    return normalized


def _normalize_vertex_graph_json(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    vertices = normalized.get("vertices")
    edges = normalized.get("edges")
    if type(vertices) is list:
        normalized["vertices"] = tuple(vertices)
    if type(edges) is list:
        normalized["edges"] = tuple(
            tuple(edge) if type(edge) is list and len(edge) == 2 else edge
            for edge in edges
        )
    return normalized


def _vertex_iso_profile_resource_estimates(
    source_order: int, source_edge_count: int, class_count: int
) -> tuple[int, int, int]:
    card_order = max(source_order - 1, 0)
    pair_count = comb(card_order, 2)
    canonical_work = (
        source_order * factorial(card_order) * (1 + card_order + pair_count)
    )
    family_check_work = source_order * (source_edge_count + card_order + 1)
    # Count the retained value's own cells: one per label character, vertex
    # entry, edge endpoint, class member, and per-vertex map position. An
    # estimated encoded size would let a consumer's encoder choice, not the
    # mathematics, decide admission.
    family_cells = source_order * (card_order + 2 * source_edge_count)
    class_cells = class_count * (1 + 32 * pair_count + 16 * card_order)
    map_cells = source_order * card_order
    return (
        canonical_work,
        canonical_work + family_check_work,
        256 + family_cells + class_cells + map_cells,
    )


def _vertex_iso_profile_value_resource_estimates(
    source_order: int, source_edge_count: int, class_count: int
) -> tuple[int, int, int]:
    """Bound wire validation, including canonicality checks on each class row."""
    card_order = max(source_order - 1, 0)
    pair_count = comb(card_order, 2)
    per_class_canonical_work = factorial(card_order) * (1 + card_order + pair_count)
    family_check_work = source_order * (source_edge_count + card_order + 1)
    map_check_work = source_order * (source_edge_count + card_order)
    _, _, output_cells = _vertex_iso_profile_resource_estimates(
        source_order, source_edge_count, class_count
    )
    return (
        class_count * per_class_canonical_work,
        class_count * per_class_canonical_work + family_check_work + map_check_work,
        output_cells,
    )


def _canonical_card_form(
    vertices: tuple[str, ...], edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[tuple[str, str], ...], tuple[int, ...]]:
    """Return the canonical edge tuple and source-position to canonical-position map."""
    n = len(vertices)
    index = {vertex: i for i, vertex in enumerate(vertices)}
    edge_indices = {frozenset((index[left], index[right])) for left, right in edges}
    pairs = tuple(combinations(range(n), 2))

    def _edge_bits(order: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            int(frozenset((order[i], order[j])) in edge_indices) for i, j in pairs
        )

    # ``permutations`` always yields the identity first, so seeding with it
    # leaves the minimum total without an absent-value sentinel.
    best_order = tuple(range(n))
    best = _edge_bits(best_order)
    for order in permutations(range(n)):
        bits = _edge_bits(order)
        if bits < best:
            best = bits
            best_order = order
    labels = tuple(f"v{i:02d}" for i in range(n))
    canonical_edges = tuple(
        (labels[i], labels[j]) for bit, (i, j) in zip(best, pairs, strict=True) if bit
    )
    source_to_canonical = [0] * n
    for canonical_position, source_position in enumerate(best_order):
        source_to_canonical[source_position] = canonical_position
    return canonical_edges, tuple(source_to_canonical)


class EdgeDeckIsomorphismProfileRequest(StrictModel):
    """Produce exact card-to-class vertex maps for a complete edge deck."""

    deck: EdgeDeletionFamily

    @model_validator(mode="before")
    @classmethod
    def preflight_raw_source_order(cls, value: Any) -> Any:
        if type(value) is not dict:
            return canonicalize_json_containers(value)
        family: Any = value.get("deck")
        source: Any = family.get("source") if type(family) is dict else None
        vertices: Any = source.get("vertices") if type(source) is dict else None
        edges: Any = source.get("edges") if type(source) is dict else None
        if type(vertices) not in (list, tuple):
            return canonicalize_json_containers(value)
        order = len(vertices)
        if order > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "edge_iso_profile_bound",
                "edge-deck isomorphism profile supports at most 10 source vertices",
            )
        pair_count = comb(order, 2)
        edge_count = len(edges) if type(edges) in (list, tuple) else pair_count
        if edge_count > pair_count:
            raise _validation_error(
                "edge_iso_profile_source_edges",
                "source edge list exceeds the simple-graph order bound",
            )
        _preflight_raw_edge_pairs(edges, pair_count)
        _preflight_edge_profile_labels(vertices, edges)
        cards: Any = family.get("cards") if type(family) is dict else None
        if type(cards) in (list, tuple):
            if len(cards) != edge_count:
                raise _validation_error(
                    "edge_iso_profile_card_count",
                    "a complete edge family must contain one card per source edge",
                )
            for card in cards:
                if type(card) is not dict:
                    continue
                if type(card.get("retained_edge_count")) is not int:
                    raise _validation_error(
                        "edge_iso_profile_integer_type",
                        "retained_edge_count must be an exact integer",
                    )
                _preflight_raw_edge_pairs(card.get("deleted_edge"), 2)
                _preflight_edge_profile_labels(card.get("deleted_edge"))
                graph: Any = card.get("card")
                if type(graph) is not dict:
                    continue
                card_vertices: Any = graph.get("vertices")
                card_edges: Any = graph.get("edges")
                retained_vertices: Any = card.get("retained_vertices")
                if (
                    (
                        type(card_vertices) in (list, tuple)
                        and len(card_vertices) > order
                    )
                    or (
                        type(retained_vertices) in (list, tuple)
                        and len(retained_vertices) > order
                    )
                    or (
                        type(card_edges) in (list, tuple)
                        and len(card_edges) > pair_count
                    )
                ):
                    raise _validation_error(
                        "edge_iso_profile_card_shape",
                        "an edge card exceeds the declared source-order shape bound",
                    )
                _preflight_raw_edge_pairs(card_edges, pair_count)
                _preflight_edge_profile_labels(
                    card_vertices,
                    card_edges,
                    retained_vertices,
                )
        _, work, output_cells = _edge_iso_profile_resource_estimates(
            order, edge_count, edge_count
        )
        if work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
            raise _validation_error(
                "edge_iso_profile_work_bound",
                "edge-deck isomorphism mapping exceeds the shared work bound",
            )
        if output_cells > MAX_EDGE_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS:
            raise _validation_error(
                "edge_iso_profile_output_bound",
                "edge-deck isomorphism profile exceeds its materialization-cell bound",
            )
        normalized = dict(value)
        normalized["deck"] = _normalize_edge_family_json(family)
        return canonicalize_json_containers(normalized)


class EdgeDeckIsomorphismClass(StrictModel):
    """One canonical class and its source-edge deletion provenance."""

    representative: SimpleUndirectedGraph
    multiplicity: int = Field(ge=1)
    card_indices: tuple[int, ...]
    deleted_edges: tuple[tuple[str, str], ...]


class EdgeDeckIsomorphismProfile(StrictModel):
    """Source-bound edge-card partition and explicit vertex bijections.

    The producer establishes canonical representatives and the exact
    isomorphism quotient. Decoding checks bounded structure, the partition,
    and the supplied bijections; it does not authenticate an authored claim
    that differently labelled representatives are nonisomorphic. To establish
    that claim, run ``edge_deck_isomorphism_profile`` on the retained family
    under its native admission rather than trusting a decoded class count.
    """

    family: EdgeDeletionFamily
    classes: tuple[EdgeDeckIsomorphismClass, ...] = Field(
        description=(
            "Canonical isomorphism classes established by the producer. "
            "Decoding validates their structural partition and maps without "
            "repeating the canonicalization search or proving class distinctness."
        )
    )
    class_indices: tuple[int, ...]
    vertex_maps: tuple[tuple[int, ...], ...]

    @model_validator(mode="before")
    @classmethod
    def admit_and_normalize_wire_value(cls, value: Any) -> Any:
        return canonicalize_json_containers(
            _admit_and_normalize_edge_iso_profile_result(value)
        )

    @model_validator(mode="after")
    def require_exact_partition_and_maps(self) -> Self:
        family = self.family
        source_order = len(family.source.vertices)
        card_count = len(family.cards)
        if source_order > MAX_UNLABELLED_DECK_VERTICES:
            raise _validation_error(
                "edge_iso_profile_bound",
                "edge-deck isomorphism profile exceeds its source-order bound",
            )
        if (
            type(self.class_indices) is not tuple
            or type(self.vertex_maps) is not tuple
            or type(self.classes) is not tuple
            or len(self.class_indices) != card_count
            or len(self.vertex_maps) != card_count
            or len(self.classes) > card_count
            or sum(item.multiplicity for item in self.classes) != card_count
        ):
            raise _validation_error(
                "edge_iso_profile_card_count",
                "profile rows and card maps must cover every source edge card",
            )
        canonical_axis = tuple(f"v{i:02d}" for i in range(source_order))
        previous_edges: tuple[tuple[str, str], ...] | None = None
        seen: list[int] = []
        for class_index, item in enumerate(self.classes):
            representative = item.representative
            if (
                type(item.card_indices) is not tuple
                or type(item.deleted_edges) is not tuple
                or not item.card_indices
                or tuple(sorted(item.card_indices)) != item.card_indices
                or item.multiplicity != len(item.card_indices)
                or len(item.deleted_edges) != item.multiplicity
                or any(
                    type(index) is not int or index < 0 or index >= card_count
                    for index in item.card_indices
                )
                or item.deleted_edges
                != tuple(
                    family.cards[index].deleted_edge for index in item.card_indices
                )
                or representative.vertices != canonical_axis
                # The representative's canonical adjacency form is the
                # producer's postcondition, established when the card was
                # canonicalized. Re-deriving it here would enumerate every
                # vertex permutation on every decode of an admitted result, so
                # result validation keeps only the structural axis, ordering,
                # and reference checks below.
                or (
                    previous_edges is not None
                    and representative.edges <= previous_edges
                )
            ):
                raise _validation_error(
                    "edge_iso_profile_class",
                    "classes must be unique, canonical, ordered, and retain exact card provenance",
                )
            previous_edges = representative.edges
            seen.extend(item.card_indices)
            for card_index in item.card_indices:
                if self.class_indices[card_index] != class_index:
                    raise _validation_error(
                        "edge_iso_profile_class_map",
                        "class_indices must agree with each class card list",
                    )
        if sorted(seen) != list(range(card_count)):
            raise _validation_error(
                "edge_iso_profile_partition",
                "class card indices must partition the edge-card axis",
            )
        for card_index, (class_index, mapping) in enumerate(
            zip(self.class_indices, self.vertex_maps, strict=True)
        ):
            if (
                type(class_index) is not int
                or class_index < 0
                or class_index >= len(self.classes)
                or type(mapping) is not tuple
                or len(mapping) != source_order
                or any(type(position) is not int for position in mapping)
                or tuple(sorted(mapping)) != tuple(range(source_order))
            ):
                raise _validation_error(
                    "edge_iso_profile_map_shape",
                    "each edge card needs a valid class index and vertex permutation",
                )
            card = family.cards[card_index].card
            positions = {
                vertex: position for position, vertex in enumerate(card.vertices)
            }
            mapped_edges = tuple(
                sorted(
                    (
                        f"v{min(mapping[positions[left]], mapping[positions[right]]):02d}",
                        f"v{max(mapping[positions[left]], mapping[positions[right]]):02d}",
                    )
                    for left, right in card.edges
                )
            )
            if mapped_edges != self.classes[class_index].representative.edges:
                raise _validation_error(
                    "edge_iso_profile_map_relation",
                    "each vertex map must carry its card edges to the class representative",
                )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


def _edge_iso_profile_resource_estimates(
    source_order: int, source_edge_count: int, class_count: int
) -> tuple[int, int, int]:
    pair_count = comb(source_order, 2)
    canonical_work = (
        source_edge_count
        * factorial(source_order)
        * (1 + source_order + 2 * pair_count)
    )
    family_check_work = source_edge_count * (source_edge_count + source_order + 1)
    map_check_work = source_edge_count * (
        source_order + max(source_edge_count - 1, 0) ** 2
    )
    class_cells = class_count * (1 + 32 * pair_count + 16 * source_order)
    family_cells = 256 * (source_order + source_edge_count * source_order)
    family_cells += 512 * (source_edge_count + source_edge_count**2)
    map_cells = source_edge_count * (16 * source_order)
    return (
        canonical_work,
        canonical_work + family_check_work + map_check_work,
        256 + family_cells + class_cells + map_cells,
    )


def _preflight_edge_profile_result_rows(
    value: dict[str, Any], order: int, pair_count: int, card_count: int
) -> None:
    class_indices: Any = value.get("class_indices")
    if type(class_indices) in (list, tuple) and len(class_indices) > card_count:
        raise _validation_error(
            "edge_iso_profile_card_count",
            "class index rows exceed the edge-card axis",
        )
    maps: Any = value.get("vertex_maps")
    if type(maps) in (list, tuple):
        if len(maps) > card_count:
            raise _validation_error(
                "edge_iso_profile_card_count",
                "vertex map rows exceed the edge-card axis",
            )
        if any(type(row) in (list, tuple) and len(row) > order for row in maps):
            raise _validation_error(
                "edge_iso_profile_map_shape",
                "a vertex map exceeds the source vertex axis",
            )
    classes: Any = value.get("classes")
    if type(classes) not in (list, tuple):
        return
    for item in classes:
        if type(item) is not dict:
            continue
        indices: Any = item.get("card_indices")
        if type(indices) in (list, tuple) and len(indices) > card_count:
            raise _validation_error(
                "edge_iso_profile_class_shape",
                "class card indices exceed the edge-card axis",
            )
        deleted_edges: Any = item.get("deleted_edges")
        if type(deleted_edges) in (list, tuple):
            if len(deleted_edges) > card_count:
                raise _validation_error(
                    "edge_iso_profile_class_shape",
                    "class deleted-edge provenance exceeds the edge-card axis",
                )
            _preflight_raw_edge_pairs(deleted_edges, card_count)
            _preflight_edge_profile_labels(deleted_edges)
        representative: Any = item.get("representative")
        if type(representative) is dict:
            vertices: Any = representative.get("vertices")
            representative_edges: Any = representative.get("edges")
            if type(vertices) in (list, tuple) and len(vertices) > order:
                raise _validation_error(
                    "edge_iso_profile_class_shape",
                    "a class representative exceeds the source vertex order",
                )
            if (
                type(representative_edges) in (list, tuple)
                and len(representative_edges) > pair_count
            ):
                raise _validation_error(
                    "edge_iso_profile_class_shape",
                    "a class representative exceeds the simple-graph edge bound",
                )
            _preflight_raw_edge_pairs(representative_edges, pair_count)
            _preflight_edge_profile_labels(vertices, representative_edges)


def _admit_and_normalize_edge_iso_profile_result(value: Any) -> Any:
    if type(value) is not dict:
        return value
    family: Any = value.get("family")
    raw_source: Any = family.get("source") if type(family) is dict else None
    vertices: Any = raw_source.get("vertices") if type(raw_source) is dict else None
    edges: Any = raw_source.get("edges") if type(raw_source) is dict else None
    if type(vertices) not in (list, tuple):
        return value
    order = len(vertices)
    if order > MAX_UNLABELLED_DECK_VERTICES:
        raise _validation_error(
            "edge_iso_profile_bound",
            "edge-deck isomorphism profile exceeds its source-order bound",
        )
    pair_count = comb(order, 2)
    edge_count = len(edges) if type(edges) in (list, tuple) else pair_count
    if edge_count > pair_count:
        raise _validation_error(
            "edge_iso_profile_source_edges",
            "source edge list exceeds the simple-graph order bound",
        )
    classes: Any = value.get("classes")
    class_count = len(classes) if type(classes) in (list, tuple) else edge_count
    if class_count > edge_count:
        raise _validation_error(
            "edge_iso_profile_class_count",
            "the profile cannot have more isomorphism classes than edge cards",
        )
    _preflight_edge_profile_family_wire(family, order, pair_count, edge_count)
    _preflight_edge_profile_result_rows(value, order, pair_count, edge_count)
    _require_exact_profile_wire_integers(value, classes, "edge")
    _, work, output_cells = _edge_iso_profile_resource_estimates(
        order, edge_count, class_count
    )
    if work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise _validation_error(
            "edge_iso_profile_validation_work_bound",
            "class representatives and card maps exceed the shared validation work bound",
        )
    if output_cells > MAX_EDGE_DECK_ISOMORPHISM_PROFILE_RESULT_CELLS:
        raise _validation_error(
            "edge_iso_profile_output_bound",
            "edge-deck isomorphism profile exceeds its materialization-cell bound",
        )
    return _normalize_edge_iso_profile_result(value)


def _preflight_raw_edge_pairs(value: Any, maximum: int) -> None:
    if type(value) not in (list, tuple):
        return
    if len(value) > maximum:
        raise _validation_error(
            "edge_iso_profile_card_shape",
            "an edge pair collection exceeds its declared shape bound",
        )
    if any(type(pair) in (list, tuple) and len(pair) > 2 for pair in value):
        raise _validation_error(
            "edge_iso_profile_card_shape",
            "each graph edge must contain at most two endpoints",
        )


def _preflight_edge_profile_family_wire(
    family: Any, order: int, pair_count: int, edge_count: int
) -> None:
    raw_cards: Any = family.get("cards") if type(family) is dict else None
    if type(raw_cards) not in (list, tuple):
        return
    if len(raw_cards) != edge_count:
        raise _validation_error(
            "edge_iso_profile_card_count",
            "a complete edge family must contain one card per source edge",
        )
    for card in raw_cards:
        if type(card) is not dict:
            continue
        if type(card.get("retained_edge_count")) is not int:
            raise _validation_error(
                "edge_iso_profile_integer_type",
                "retained_edge_count must be an exact integer",
            )
        _preflight_raw_edge_pairs(card.get("deleted_edge"), 2)
        _preflight_edge_profile_labels(card.get("deleted_edge"))
        graph: Any = card.get("card")
        if type(graph) is not dict:
            continue
        card_vertices: Any = graph.get("vertices")
        card_edges: Any = graph.get("edges")
        retained_vertices: Any = card.get("retained_vertices")
        if (
            (type(card_vertices) in (list, tuple) and len(card_vertices) > order)
            or (
                type(retained_vertices) in (list, tuple)
                and len(retained_vertices) > order
            )
            or (type(card_edges) in (list, tuple) and len(card_edges) > pair_count)
        ):
            raise _validation_error(
                "edge_iso_profile_card_shape",
                "an edge card exceeds the declared source-order shape bound",
            )
        _preflight_raw_edge_pairs(card_edges, pair_count)
        _preflight_edge_profile_labels(card_vertices, card_edges, retained_vertices)


def _preflight_edge_profile_labels(*collections: Any) -> None:
    for collection in collections:
        if type(collection) not in (list, tuple):
            continue
        for item in collection:
            labels = item if type(item) in (list, tuple) else (item,)
            if any(
                type(label) is str and len(label) > MAX_GRAPH_LABEL_BYTES
                for label in labels
            ):
                raise _validation_error(
                    "edge_iso_profile_label_bound",
                    "graph labels exceed the 64-byte scalar bound",
                )


def _normalize_edge_family_json(value: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_vertex_family_json(value)
    cards: Any = normalized.get("cards")
    if type(cards) is tuple:
        rows = []
        for item in cards:
            if type(item) is not dict:
                rows.append(item)
                continue
            row = dict(item)
            deleted = row.get("deleted_edge")
            if type(deleted) is list:
                row["deleted_edge"] = tuple(deleted)
            rows.append(row)
        normalized["cards"] = tuple(rows)
    return normalized


def _require_exact_profile_wire_integers(
    value: dict[str, Any], classes: Any, kind: str
) -> None:
    class_indices: Any = value.get("class_indices")
    if type(class_indices) in (list, tuple) and any(
        type(index) is not int for index in class_indices
    ):
        raise _validation_error(
            f"{kind}_iso_profile_integer_type",
            "class_indices must contain exact integers",
        )
    vertex_maps: Any = value.get("vertex_maps")
    if type(vertex_maps) in (list, tuple) and any(
        type(row) in (list, tuple)
        and any(type(position) is not int for position in row)
        for row in vertex_maps
    ):
        raise _validation_error(
            f"{kind}_iso_profile_integer_type",
            "vertex_maps must contain exact integers",
        )
    if type(classes) not in (list, tuple):
        return
    for item in classes:
        if type(item) is not dict:
            continue
        if type(item.get("multiplicity")) is not int:
            raise _validation_error(
                f"{kind}_iso_profile_integer_type",
                "class multiplicity must be an exact integer",
            )
        card_indices: Any = item.get("card_indices")
        if type(card_indices) in (list, tuple) and any(
            type(index) is not int for index in card_indices
        ):
            raise _validation_error(
                f"{kind}_iso_profile_integer_type",
                "class card_indices must contain exact integers",
            )


def _normalize_edge_iso_profile_result(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    family: Any = normalized.get("family")
    if type(family) is dict:
        normalized["family"] = _normalize_edge_family_json(family)
    classes: Any = normalized.get("classes")
    if type(classes) is list:
        rows = []
        for item in classes:
            if type(item) is not dict:
                rows.append(item)
                continue
            row = dict(item)
            for field in ("card_indices", "deleted_edges"):
                if type(row.get(field)) is list:
                    row[field] = tuple(
                        tuple(entry) if field == "deleted_edges" else entry
                        for entry in row[field]
                    )
            representative = row.get("representative")
            if type(representative) is dict:
                row["representative"] = _normalize_vertex_graph_json(representative)
            rows.append(row)
        normalized["classes"] = tuple(rows)
    for field in ("class_indices", "vertex_maps"):
        raw_rows: Any = normalized.get(field)
        if type(raw_rows) is list:
            normalized[field] = tuple(
                tuple(row) if field == "vertex_maps" and type(row) is list else row
                for row in raw_rows
            )
    return normalized
