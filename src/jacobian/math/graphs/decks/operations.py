"""Exact native kernels for source-bound vertex-deletion families."""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb, factorial

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_DECK_CARD_EDGES,
    MAX_DECK_ECHO_ALLOCATION,
    MAX_DECK_VERTICES,
    MAX_DEGREE_MULTISET_DIGITS,
    MAX_EDGE_DECK_EDGES,
    MAX_KELLY_DECK_TOTAL_WORK,
    MAX_UNLABELLED_DECK_ISOMORPHISM_WORK,
    MAX_UNLABELLED_DECK_VERTICES,
    EdgeDeletionFamily,
    SourceBoundEdgeCard,
    SourceBoundVertexCard,
    UnlabelledDeck,
    UnlabelledDeckClass,
    UnlabelledVertexDeck,
    UnlabelledVertexDeckClass,
    VertexDeckEdgeCount,
    VertexDeckInducedSubgraphContribution,
    VertexDeckInducedSubgraphCount,
    VertexDeckSubgraphContribution,
    VertexDeckSubgraphCount,
    VertexDeletionFamily,
)
from jacobian.math.graphs.patterns._models import _require_bounded_request
from jacobian.math.graphs.patterns.operations import (
    induced_vertex_subset_pattern_count,
)
from jacobian.math.graphs.realization._models import DegreeSequence
from jacobian.math.graphs.values import SimpleUndirectedGraph

__all__ = [
    "edge_deletion_family",
    "unlabelled_deck",
    "unlabelled_vertex_deck",
    "verify_edge_deletion_family",
    "verify_vertex_deletion_family",
    "vertex_deck_degree_multiset",
    "vertex_deck_edge_count",
    "vertex_deck_induced_subgraph_count",
    "vertex_deck_subgraph_count",
    "vertex_deletion_family",
]


def _admit_deck_graph(graph: SimpleUndirectedGraph) -> SimpleUndirectedGraph:
    """Shared deck admission for native and catalog callers.

    Bounds the source order and the aggregate card-edge output before any
    card materialization.
    """
    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph_deck.graph_carrier",
            message="vertex decks require a SimpleUndirectedGraph value",
        )
    order = len(graph.vertices)
    if order > MAX_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.vertex_bound",
            message=(
                "vertex-deletion decks support at most "
                f"{MAX_DECK_VERTICES} source vertices"
            ),
        )
    edge_count = len(graph.edges)
    if order >= 2 and (order - 2) * edge_count > MAX_DECK_CARD_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.card_edge_bound",
            message="aggregate card edges exceed the exact output bound",
        )
    return graph


def _admit_edge_deck_graph(graph: SimpleUndirectedGraph) -> SimpleUndirectedGraph:
    if type(graph) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("graph",),
            code="graph_deck.graph_carrier",
            message="edge decks require a SimpleUndirectedGraph value",
        )
    edge_count = len(graph.edges)
    if edge_count * max(edge_count - 1, 0) > MAX_EDGE_DECK_EDGES:
        raise OperationResourceAdmissionError(
            location=("graph",),
            code="graph_deck.edge_card_edge_bound",
            message="aggregate edge-card output exceeds the exact bound",
        )
    return graph


def edge_deletion_family(graph: SimpleUndirectedGraph) -> EdgeDeletionFamily:
    """Return every source-edge-deleted card, retaining isolated vertices."""
    source = _admit_edge_deck_graph(graph)
    cards: list[SourceBoundEdgeCard] = []
    for deleted in source.edges:
        card_edges = tuple(edge for edge in source.edges if edge != deleted)
        cards.append(
            SourceBoundEdgeCard.model_construct(
                deleted_edge=deleted,
                card=SimpleUndirectedGraph(vertices=source.vertices, edges=card_edges),
                retained_vertices=source.vertices,
                retained_edge_count=len(card_edges),
            )
        )
    return EdgeDeletionFamily._from_kernel(source=source, cards=tuple(cards))


def verify_edge_deletion_family(claim: EdgeDeletionFamily) -> bool:
    if type(claim) is not EdgeDeletionFamily:
        return False
    return edge_deletion_family(claim.source) == claim


def _admit_edge_deletion_family(family: EdgeDeletionFamily) -> EdgeDeletionFamily:
    """Re-establish the exact source-minus-edge relation at a consumer boundary."""
    if type(family) is not EdgeDeletionFamily:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.family_carrier",
            message="deck must be an EdgeDeletionFamily",
        )
    source = family.source
    _admit_edge_deck_graph(source)
    if type(family.cards) is not tuple or len(family.cards) != len(source.edges):
        raise OperationDomainValidationError(
            location=("deck", "cards"),
            code="graph_deck.family_card_count",
            message="edge deletion family must contain one card per source edge",
        )
    for expected_edge, card in zip(source.edges, family.cards, strict=True):
        if type(card) is not SourceBoundEdgeCard:
            raise OperationDomainValidationError(
                location=("deck", "cards"),
                code="graph_deck.card_carrier",
                message="edge deletion family cards must use the canonical card type",
            )
        if (
            card.deleted_edge != expected_edge
            or type(card.card) is not SimpleUndirectedGraph
            or card.retained_vertices != source.vertices
            or card.card.vertices != source.vertices
            or card.card.edges
            != tuple(edge for edge in source.edges if edge != expected_edge)
            or card.retained_edge_count != len(card.card.edges)
        ):
            raise OperationDomainValidationError(
                location=("deck", "cards"),
                code="graph_deck.card_deletion_relation",
                message="each edge card must equal the source graph with its bound edge removed",
            )
    return family


def unlabelled_deck(family: EdgeDeletionFamily) -> UnlabelledDeck:
    """Quotient an edge deck into exact unlabeled isomorphism classes.

    Each card is assigned the least adjacency bit word over all vertex
    permutations. Equal words are equivalent exactly under graph isomorphism;
    unlike pairwise similarity searches, this key is a complete invariant, so
    the aggregate permutation work is admitted before any canonical form is
    computed.
    """
    family = _admit_edge_deletion_family(family)
    source_order = len(family.source.vertices)
    card_count = len(family.cards)
    if source_order > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.isomorphism_vertex_bound",
            message="unlabelled quotient exceeds the isomorphism envelope",
        )
    work = (
        card_count
        * factorial(source_order)
        * (1 + source_order + comb(source_order, 2))
    )
    if work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.isomorphism_work_bound",
            message="unlabelled quotient exceeds the exact permutation work envelope",
        )
    classes: list[UnlabelledDeckClass] = []
    signatures: dict[int, int] = {}
    for index, card in enumerate(family.cards):
        signature = _canonical_adjacency_signature(card.card)
        class_index = signatures.get(signature)
        if class_index is None:
            signatures[signature] = len(classes)
            classes.append(
                UnlabelledDeckClass.model_construct(
                    representative=card.card, multiplicity=1, card_indices=(index,)
                )
            )
        else:
            item = classes[class_index]
            classes[class_index] = item.model_copy(
                update={
                    "multiplicity": item.multiplicity + 1,
                    "card_indices": (*item.card_indices, index),
                }
            )
    return UnlabelledDeck._from_kernel(
        source=family.source, classes=tuple(classes), card_count=card_count
    )


def unlabelled_vertex_deck(family: VertexDeletionFamily) -> UnlabelledVertexDeck:
    """Group vertex-deleted cards into exact isomorphism classes.

    The family is reconstructed from its source before quotienting, so a
    caller-created model bypass cannot omit, duplicate, or alter cards. The
    exact permutation enumeration work is admitted before any canonical forms
    are computed.
    """
    if type(family) is not VertexDeletionFamily:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.vertex_family_carrier",
            message="deck must be a VertexDeletionFamily",
        )
    source = _admit_deck_graph(family.source)
    n = len(source.vertices)
    if n > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.vertex_isomorphism_bound",
            message="unlabelled vertex deck exceeds the isomorphism envelope",
        )
    card_order = max(n - 1, 0)
    work = n * factorial(card_order) * (1 + card_order + comb(card_order, 2))
    if work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.vertex_isomorphism_work_bound",
            message="unlabelled vertex deck exceeds the exact permutation work bound",
        )
    if vertex_deletion_family(source) != family:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.vertex_family_relation",
            message="vertex deletion family must equal the complete source family",
        )
    classes: list[UnlabelledVertexDeckClass] = []
    signatures: dict[int, int] = {}
    for index, card in enumerate(family.cards):
        signature = _canonical_adjacency_signature(card.card)
        if signature in signatures:
            class_index = signatures[signature]
            item = classes[class_index]
            classes[class_index] = item.model_copy(
                update={
                    "multiplicity": item.multiplicity + 1,
                    "card_indices": (*item.card_indices, index),
                }
            )
        else:
            signatures[signature] = len(classes)
            classes.append(
                UnlabelledVertexDeckClass.model_construct(
                    representative=card.card,
                    multiplicity=1,
                    card_indices=(index,),
                )
            )
    return UnlabelledVertexDeck._from_kernel(
        family=family, classes=tuple(classes), card_count=len(family.cards)
    )


def _canonical_adjacency_signature(graph: SimpleUndirectedGraph) -> int:
    """Return the least adjacency bit word over every vertex ordering."""
    vertices = graph.vertices
    edges = {frozenset(edge) for edge in graph.edges}
    least: int | None = None
    for ordering in permutations(vertices):
        code = 0
        for i in range(len(ordering)):
            left = ordering[i]
            for j in range(i + 1, len(ordering)):
                code = (code << 1) | int(frozenset((left, ordering[j])) in edges)
        if least is None or code < least:
            least = code
    return 0 if least is None else least


def _require_kelly_echo_allocation(
    source: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
    *,
    code: str,
) -> None:
    """Bound the label allocation a Kelly count result can echo.

    The result embeds the source family, one class representative per deck
    class, and one contribution representative per class: at most ``3n + 1``
    graph echoes over ``n`` source vertices. Each label is echoed at most once
    per incident record, so one graph echo allocates at most ``n`` copies of
    the aggregate source label characters. Ledger, index, and multiplicity
    integers are cardinality-bounded by ``n``, and count digits are bounded by
    the result encodings, so this bound governs the only unbounded growth.
    """
    source_order = len(source.vertices)
    pattern_order = len(pattern.vertices)
    source_characters = sum(len(label) for label in source.vertices)
    pattern_characters = sum(len(label) for label in pattern.vertices)
    echo_allocation = (3 * source_order + 1) * max(
        source_order, 1
    ) * source_characters + (pattern_order + 1) * pattern_characters
    if echo_allocation > MAX_DECK_ECHO_ALLOCATION:
        raise OperationResourceAdmissionError(
            location=("deck", "pattern"),
            code=code,
            message="Kelly deck count exceeds the exact result echo allocation bound",
        )


def vertex_deck_induced_subgraph_count(
    deck: UnlabelledVertexDeck,
    pattern: SimpleUndirectedGraph,
) -> VertexDeckInducedSubgraphCount:
    """Reconstruct an induced subset count by Kelly's double-counting identity.

    For pattern order ``h < n``, each induced copy survives in exactly ``n-h``
    vertex cards. Card counts follow the existing exact induced vertex-subset
    operation, not a non-induced embedding convention.
    """
    if type(deck) is not UnlabelledVertexDeck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_deck_carrier",
            message="deck must be an UnlabelledVertexDeck",
        )
    if type(pattern) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph_deck.kelly_pattern_carrier",
            message="pattern must be a SimpleUndirectedGraph",
        )
    family = deck.family
    source = _admit_deck_graph(family.source)
    source_order = len(source.vertices)
    pattern_order = len(pattern.vertices)
    if pattern_order >= source_order:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph_deck.kelly_pattern_not_proper",
            message="the pattern order must be strictly smaller than source order",
        )
    if source_order > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.kelly_vertex_bound",
            message="Kelly count exceeds the vertex-deck isomorphism envelope",
        )
    if (
        type(family.cards) is not tuple
        or len(family.cards) != source_order
        or any(
            type(card) is not SourceBoundVertexCard
            or type(card.card) is not SimpleUndirectedGraph
            for card in family.cards
        )
        or type(deck.classes) is not tuple
        or len(deck.classes) > source_order
        or any(
            type(card_class) is not UnlabelledVertexDeckClass
            or type(card_class.representative) is not SimpleUndirectedGraph
            or type(card_class.card_indices) is not tuple
            or len(card_class.card_indices) > source_order
            for card_class in deck.classes
        )
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_value_shape",
            message="vertex deck has an invalid bounded family or class shape",
        )

    card_order = source_order - 1
    quotient_work = source_order * factorial(card_order) * (
        1 + card_order + comb(card_order, 2)
    ) + 2 * max(source_order - 2, 0) * len(source.edges)
    pattern_work = 0
    for card in family.cards:
        try:
            pattern_work += _require_bounded_request(card.card, pattern)
        except PydanticCustomError as exc:
            raise OperationResourceAdmissionError(
                location=("pattern",),
                code=exc.type,
                message=exc.message(),
            ) from None
    family_work = source_order * (len(source.vertices) + len(source.edges)) + (
        max(source_order - 2, 0) * len(source.edges)
    )
    if quotient_work + pattern_work + family_work > MAX_KELLY_DECK_TOTAL_WORK:
        raise OperationResourceAdmissionError(
            location=("deck", "pattern"),
            code="graph_deck.kelly_total_work_bound",
            message=(
                "Kelly deck canonicalization and induced-pattern counts exceed "
                f"the {MAX_KELLY_DECK_TOTAL_WORK:,}-unit total work bound"
            ),
        )
    _require_kelly_echo_allocation(
        source, pattern, code="graph_deck.kelly_result_allocation_bound"
    )

    # This replays the complete family and exact class quotient before relying
    # on caller-supplied card multiplicities.
    canonical_deck = unlabelled_vertex_deck(family)
    if canonical_deck != deck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_class_partition",
            message="vertex deck classes must be the exact multiset quotient of its family",
        )

    contributions: list[VertexDeckInducedSubgraphContribution] = []
    weighted_total = 0
    for class_index, card_class in enumerate(canonical_deck.classes):
        per_card_count = induced_vertex_subset_pattern_count(
            card_class.representative, pattern
        )
        weighted = per_card_count * card_class.multiplicity
        weighted_total += weighted
        contributions.append(
            VertexDeckInducedSubgraphContribution.model_construct(
                class_index=class_index,
                representative_card=card_class.representative,
                card_indices=card_class.card_indices,
                multiplicity=card_class.multiplicity,
                occurrences_per_card=per_card_count,
                weighted_occurrences=weighted,
            )
        )

    divisor = source_order - pattern_order
    if weighted_total % divisor:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_nondivisible",
            message="complete source-bound deck violates Kelly's divisibility identity",
        )
    return VertexDeckInducedSubgraphCount._from_kernel(
        deck=canonical_deck,
        pattern=pattern,
        contributions=tuple(contributions),
        weighted_card_total=weighted_total,
        overcount_divisor=divisor,
        occurrence_count=weighted_total // divisor,
    )


def _noninduced_copy_count(
    host: SimpleUndirectedGraph,
    pattern: SimpleUndirectedGraph,
    pattern_edges: tuple[tuple[str, str], ...],
    pattern_automorphisms: int,
) -> int:
    """Count edge-subset copies, quotienting injective embeddings by Aut(F)."""
    pattern_order = len(pattern.vertices)
    host_edges = {frozenset(edge) for edge in host.edges}
    embeddings = 0
    for selected in combinations(host.vertices, pattern_order):
        for image in permutations(selected):
            if all(
                frozenset(
                    (
                        image[pattern.vertices.index(left)],
                        image[pattern.vertices.index(right)],
                    )
                )
                in host_edges
                for left, right in pattern_edges
            ):
                embeddings += 1
    if embeddings % pattern_automorphisms:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph_deck.subgraph_embedding_quotient",
            message="injective subgraph embeddings must divide evenly by the pattern automorphism count",
        )
    return embeddings // pattern_automorphisms


def vertex_deck_subgraph_count(
    deck: UnlabelledVertexDeck,
    pattern: SimpleUndirectedGraph,
) -> VertexDeckSubgraphCount:
    """Count ordinary subgraph copies from a complete vertex deck.

    The copy convention is an edge subset isomorphic to the pattern on a
    vertex subset; extra host edges are permitted. Kelly's identity gives the
    source count by dividing the multiplicity-weighted card counts by n-k.
    """
    if (
        type(deck) is not UnlabelledVertexDeck
        or type(pattern) is not SimpleUndirectedGraph
    ):
        raise OperationDomainValidationError(
            location=("deck", "pattern"),
            code="graph_deck.kelly_subgraph_carrier",
            message="deck and pattern must have their canonical graph carriers",
        )
    family = deck.family
    source = _admit_deck_graph(family.source)
    try:
        source = SimpleUndirectedGraph.model_validate(source.model_dump())
        pattern = SimpleUndirectedGraph.model_validate(pattern.model_dump())
    except (ValidationError, TypeError, ValueError):
        raise OperationDomainValidationError(
            location=("deck", "pattern"),
            code="graph_deck.kelly_subgraph_graph_values",
            message="source and pattern must be canonical simple graph values",
        ) from None
    n, k = len(source.vertices), len(pattern.vertices)
    if k >= n:
        raise OperationDomainValidationError(
            location=("pattern",),
            code="graph_deck.kelly_subgraph_pattern_not_proper",
            message="the pattern order must be strictly smaller than source order",
        )
    if n > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_vertex_bound",
            message="Kelly subgraph count exceeds the vertex-deck isomorphism envelope",
        )
    if (
        type(family.cards) is not tuple
        or len(family.cards) != n
        or any(
            type(card) is not SourceBoundVertexCard
            or type(card.card) is not SimpleUndirectedGraph
            or type(card.card.vertices) is not tuple
            or type(card.card.edges) is not tuple
            or type(card.retained_vertices) is not tuple
            or len(card.card.vertices) > max(n - 1, 0)
            or len(card.card.edges) > comb(max(n - 1, 0), 2)
            or len(card.retained_vertices) > max(n - 1, 0)
            for card in family.cards
        )
        or type(deck.classes) is not tuple
        or len(deck.classes) > n
        or any(
            type(card_class) is not UnlabelledVertexDeckClass
            or type(card_class.representative) is not SimpleUndirectedGraph
            or type(card_class.representative.vertices) is not tuple
            or type(card_class.representative.edges) is not tuple
            or len(card_class.representative.vertices) > max(n - 1, 0)
            or len(card_class.representative.edges) > comb(max(n - 1, 0), 2)
            or type(card_class.card_indices) is not tuple
            or len(card_class.card_indices) > n
            for card_class in deck.classes
        )
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_value_shape",
            message="vertex deck has an invalid bounded family or class shape",
        )
    pattern_edges = pattern.edges
    pattern_edge_set = {frozenset(edge) for edge in pattern_edges}
    # Admission charges each source card's candidate injections and edge
    # probes, all pattern automorphism candidates, family binding, and deck
    # canonicalization before count expansion.
    assignments_per_card = comb(n - 1, k) * factorial(k)
    pattern_work = n * assignments_per_card * max(1, k * len(pattern_edges))
    pattern_work += factorial(k) * max(1, k * len(pattern_edges))
    quotient_work = n * factorial(n - 1) * (1 + (n - 1) + comb(n - 1, 2))
    family_work = n * (n + len(source.edges)) + max(n - 2, 0) * len(source.edges)
    total_work = pattern_work + quotient_work + family_work
    if quotient_work > MAX_UNLABELLED_DECK_ISOMORPHISM_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_quotient_work_bound",
            message="Kelly subgraph counting exceeds the exact vertex-deck quotient work bound",
        )
    if total_work > MAX_KELLY_DECK_TOTAL_WORK:
        raise OperationResourceAdmissionError(
            location=("deck", "pattern"),
            code="graph_deck.kelly_subgraph_total_work_bound",
            message=(
                "Kelly deck canonicalization and ordinary subgraph counts exceed "
                f"the {MAX_KELLY_DECK_TOTAL_WORK:,}-unit total work bound"
            ),
        )
    _require_kelly_echo_allocation(
        source, pattern, code="graph_deck.kelly_subgraph_result_allocation"
    )

    # Validation and quotient construction occur only after aggregate work and
    # output have been admitted. The budget includes these complete passes.
    if vertex_deletion_family(source) != family:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_family_relation",
            message="deck must retain the exact complete source-bound deletion family",
        )
    canonical = unlabelled_vertex_deck(family)
    if canonical != deck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_class_partition",
            message="vertex deck classes must be the exact multiset quotient of its family",
        )

    automorphisms = sum(
        all(
            frozenset(
                (
                    image[pattern.vertices.index(left)],
                    image[pattern.vertices.index(right)],
                )
            )
            in pattern_edge_set
            for left, right in pattern_edges
        )
        for image in permutations(pattern.vertices)
    )

    contributions: list[VertexDeckSubgraphContribution] = []
    weighted_total = 0
    for class_index, card_class in enumerate(canonical.classes):
        per_card = _noninduced_copy_count(
            card_class.representative, pattern, pattern_edges, automorphisms
        )
        weighted = per_card * card_class.multiplicity
        weighted_total += weighted
        contributions.append(
            VertexDeckSubgraphContribution.model_construct(
                class_index=class_index,
                representative_card=card_class.representative,
                card_indices=card_class.card_indices,
                multiplicity=card_class.multiplicity,
                occurrences_per_card=per_card,
                weighted_occurrences=weighted,
            )
        )
    divisor = n - k
    if weighted_total % divisor:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.kelly_subgraph_nondivisible",
            message="complete source-bound deck violates Kelly's subgraph divisibility identity",
        )
    return VertexDeckSubgraphCount._from_kernel(
        deck=canonical,
        pattern=pattern,
        contributions=tuple(contributions),
        weighted_card_total=weighted_total,
        overcount_divisor=divisor,
        occurrence_count=weighted_total // divisor,
    )


def vertex_deck_edge_count(
    deck: UnlabelledVertexDeck,
) -> VertexDeckEdgeCount:
    """Recover ``|E(G)|`` from the complete vertex-deck edge-count multiset.

    For an n-vertex graph with n >= 3, each source edge occurs in exactly
    n-2 vertex-deleted cards. The quotient is computed from deck classes,
    never from the source graph's edge count.
    """
    if type(deck) is not UnlabelledVertexDeck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.edge_count_carrier",
            message="deck must be an UnlabelledVertexDeck",
        )
    family = deck.family
    if type(family) is not VertexDeletionFamily:
        raise OperationDomainValidationError(
            location=("deck", "family"),
            code="graph_deck.edge_count_family_carrier",
            message="deck must retain a complete source-bound vertex family",
        )
    source = _admit_deck_graph(family.source)
    source_order = len(source.vertices)
    if source_order < 3:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.edge_count_order",
            message="vertex-deck edge count requires at least three source vertices",
        )
    if source_order > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.edge_count_vertex_bound",
            message="edge-count reconstruction exceeds the unlabelled deck envelope",
        )
    if (
        type(family.cards) is not tuple
        or len(family.cards) != source_order
        or type(deck.classes) is not tuple
        or len(deck.classes) > source_order
        or any(
            type(card_class) is not UnlabelledVertexDeckClass
            or type(card_class.representative) is not SimpleUndirectedGraph
            or type(card_class.card_indices) is not tuple
            or len(card_class.card_indices) > source_order
            for card_class in deck.classes
        )
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.edge_count_shape",
            message="vertex deck has an invalid bounded family or class shape",
        )

    card_order = source_order - 1
    quotient_work = (
        source_order * factorial(card_order) * (1 + card_order + comb(card_order, 2))
    )
    family_work = source_order * (len(source.vertices) + len(source.edges)) + (
        max(source_order - 2, 0) * len(source.edges)
    )
    if quotient_work + family_work > MAX_KELLY_DECK_TOTAL_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.edge_count_work_bound",
            message="deck validation and edge-count reconstruction exceed the aggregate work bound",
        )

    # Caller-supplied classes are claims. Rebuild the exact source family and
    # isomorphism quotient before relying on their multiplicities.
    canonical_deck = unlabelled_vertex_deck(family)
    if canonical_deck != deck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.edge_count_class_partition",
            message="vertex deck classes must be the exact multiset quotient of its family",
        )

    card_edge_counts = tuple(
        sorted(
            len(card_class.representative.edges)
            for card_class in canonical_deck.classes
            for _ in range(card_class.multiplicity)
        )
    )
    card_edge_total = sum(
        len(card_class.representative.edges) * card_class.multiplicity
        for card_class in canonical_deck.classes
    )
    divisor = source_order - 2
    if card_edge_total % divisor:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.edge_count_nondivisible",
            message="complete vertex deck violates the source-edge divisibility identity",
        )
    return VertexDeckEdgeCount._from_kernel(
        deck=canonical_deck,
        card_edge_counts=card_edge_counts,
        card_edge_total=card_edge_total,
        overcount_divisor=divisor,
        source_edge_count=card_edge_total // divisor,
    )


def vertex_deck_degree_multiset(
    deck: UnlabelledVertexDeck,
) -> DegreeSequence:
    """Recover the source degree multiset from its complete vertex deck.

    Each card G-v has exactly m-deg(v) edges. The edge-count operation first
    authenticates the source-bound family and exact multiset quotient, then
    computes m from the cards using the vertex-deck identity. Thus this
    projection relies only on card edge counts and class multiplicities; the
    source graph's degree data is never consulted.
    """
    if type(deck) is not UnlabelledVertexDeck:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.degree_multiset_carrier",
            message="deck must be an UnlabelledVertexDeck",
        )
    if type(deck.family) is not VertexDeletionFamily:
        raise OperationDomainValidationError(
            location=("deck", "family"),
            code="graph_deck.degree_multiset_family_carrier",
            message="deck must retain a complete source-bound vertex family",
        )
    if type(deck.family.source) is not SimpleUndirectedGraph:
        raise OperationDomainValidationError(
            location=("deck", "family", "source"),
            code="graph_deck.degree_multiset_source_carrier",
            message="deck source must be a SimpleUndirectedGraph",
        )
    source_order = len(deck.family.source.vertices)
    if source_order < 3:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.degree_multiset_order",
            message="vertex-deck degree multiset requires at least three vertices",
        )
    if source_order > MAX_UNLABELLED_DECK_VERTICES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.degree_multiset_output_bound",
            message="degree multiset output exceeds the exact deck envelope",
        )
    degree_digit_bound = source_order * len(str(source_order - 1))
    if degree_digit_bound > MAX_DEGREE_MULTISET_DIGITS:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.degree_multiset_result_digits",
            message="degree multiset exceeds its exact decimal digit bound",
        )

    # The existing operation admits all family validation and exact
    # canonicalization work before materializing or trusting card multiplicity.
    edge_count = vertex_deck_edge_count(deck)
    return DegreeSequence(
        degrees=tuple(
            sorted(
                (
                    edge_count.source_edge_count - card_edge_count
                    for card_edge_count in edge_count.card_edge_counts
                ),
                reverse=True,
            )
        )
    )


def _delete_vertex(
    graph: SimpleUndirectedGraph, deleted: str
) -> tuple[SimpleUndirectedGraph, int, int]:
    """Return the card of ``deleted`` plus retained/deleted edge counts."""
    retained = tuple(vertex for vertex in graph.vertices if vertex != deleted)
    retained_set = set(retained)
    card_edges = tuple(
        edge
        for edge in graph.edges
        if edge[0] in retained_set and edge[1] in retained_set
    )
    card = SimpleUndirectedGraph(vertices=retained, edges=card_edges)
    return card, len(card_edges), len(graph.edges) - len(card_edges)


def vertex_deletion_family(graph: SimpleUndirectedGraph) -> VertexDeletionFamily:
    """Build the complete source-bound vertex-deletion family of ``graph``.

    Iterates the exact source vertex domain once, deletes each vertex
    directly, binds every retained vertex and edge to the source, then
    uses the Kelly counting identities: every source edge appears in
    exactly ``n-2`` cards (for ``n >= 2``) and every source vertex in
    exactly ``n-1`` card domains.
    """
    source = _admit_deck_graph(graph)
    order = len(source.vertices)
    cards: list[SourceBoundVertexCard] = []
    for deleted in source.vertices:
        card, retained_count, deleted_count = _delete_vertex(source, deleted)
        cards.append(
            SourceBoundVertexCard.model_construct(
                deleted_vertex=deleted,
                card=card,
                retained_vertices=card.vertices,
                retained_edge_count=retained_count,
                deleted_edge_count=deleted_count,
            )
        )
    edge_appearances = (max(order - 2, 0),) * len(source.edges)
    vertex_appearances = (max(order - 1, 0),) * order
    return VertexDeletionFamily._from_kernel(
        source,
        tuple(cards),
        edge_appearances,
        vertex_appearances,
    )


def verify_vertex_deletion_family(claim: VertexDeletionFamily) -> bool:
    """Replay deletion and counting identities for a serialized family."""
    try:
        return vertex_deletion_family(claim.source) == claim
    except (OperationDomainValidationError, TypeError, ValueError):
        return False
