"""Exact bounded cardwise connected-component profiles for graph decks."""

from collections import Counter, deque
from math import comb

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_ANONYMOUS_CARD_CLASSES,
    MAX_ANONYMOUS_CARD_RESULT_UNITS,
    MAX_UNLABELLED_DECK_VERTICES,
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
)
from jacobian.math.graphs.decks.card_component_profile._models import (
    AnonymousDeckComponentProfile,
    CardComponentSizeProfile,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_CARD_COMPONENT_PROFILE_WORK = 2_000_000
MAX_CARD_COMPONENT_PROFILE_OUTPUT_BYTES = 1_000_000
_PARTITION_COUNTS_THROUGH_TEN = (1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42)


def _admit_deck(
    deck: object,
) -> tuple[int, tuple[tuple[tuple[tuple[str, str], ...], int], ...], int]:
    if type(deck) is not AnonymousGraphCardMultiset:
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.component_profile_carrier",
            message="deck must be an AnonymousGraphCardMultiset value",
        )
    order = getattr(deck, "card_order", None)
    classes = getattr(deck, "classes", None)
    if (
        type(order) is not int
        or order < 0
        or order > MAX_UNLABELLED_DECK_VERTICES
        or type(classes) is not tuple
        or len(classes) > MAX_ANONYMOUS_CARD_CLASSES
    ):
        raise OperationDomainValidationError(
            location=("deck",),
            code="graph_deck.component_profile_shape",
            message="anonymous deck has a malformed bounded shape",
        )

    pair_count = comb(order, 2)
    input_bytes = len(classes) * (64 + 16 * pair_count)
    profile_rows = min(len(classes), _PARTITION_COUNTS_THROUGH_TEN[order])
    output_bytes = 128 + profile_rows * (64 + 4 * order)
    if input_bytes > MAX_ANONYMOUS_CARD_RESULT_UNITS:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.component_profile_input_bound",
            message="deck class input exceeds the canonical representation unit bound",
        )
    if output_bytes > MAX_CARD_COMPONENT_PROFILE_OUTPUT_BYTES:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.component_profile_output_bound",
            message="component profile output exceeds its serialized byte bound",
        )

    expected_vertices = tuple(f"v{i:02d}" for i in range(order))
    rows: list[tuple[tuple[tuple[str, str], ...], int]] = []
    total_card_count = 0
    for index, item in enumerate(classes):
        if type(item) is not AnonymousGraphCardClass:
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.component_profile_class_carrier",
                message="deck classes must use AnonymousGraphCardClass",
            )
        graph = getattr(item, "representative", None)
        multiplicity = getattr(item, "multiplicity", None)
        if type(graph) is not SimpleUndirectedGraph:
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.component_profile_class",
                message="each class needs a fixed-axis graph and positive bounded multiplicity",
            )
        # Only inspect fields on the exact graph carrier; a forged arbitrary
        # object could expose hostile or unbounded descriptors here.
        vertices = getattr(graph, "vertices", None)
        edges = getattr(graph, "edges", None)
        if (
            type(vertices) is not tuple
            or vertices != expected_vertices
            or type(edges) is not tuple
            or len(edges) > pair_count
            or type(multiplicity) is not int
            or not 1 <= multiplicity < 10**12
        ):
            raise OperationDomainValidationError(
                location=("deck", "classes", index),
                code="graph_deck.component_profile_class",
                message="each class needs a fixed-axis graph and positive bounded multiplicity",
            )
        for edge in edges:
            if (
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(label) is not str or len(label) > 3 for label in edge)
                or edge[0] not in expected_vertices
                or edge[1] not in expected_vertices
                or edge[0] >= edge[1]
            ):
                raise OperationDomainValidationError(
                    location=("deck", "classes", index, "representative", "edges"),
                    code="graph_deck.component_profile_edges",
                    message="card edges must be canonical pairs on the declared axis",
                )
        if tuple(sorted(set(edges))) != edges:
            raise OperationDomainValidationError(
                location=("deck", "classes", index, "representative", "edges"),
                code="graph_deck.component_profile_edges",
                message="card edges must be unique and ordered",
            )
        rows.append((edges, multiplicity))
        total_card_count += multiplicity

    connectivity_work = len(classes) * (order + 2 * pair_count)
    if connectivity_work > MAX_CARD_COMPONENT_PROFILE_WORK:
        raise OperationResourceAdmissionError(
            location=("deck",),
            code="graph_deck.component_profile_work_bound",
            message="card component analysis exceeds the exact work bound",
        )
    return order, tuple(rows), total_card_count


def _component_orders(
    order: int, edges: tuple[tuple[str, str], ...]
) -> tuple[int, ...]:
    adjacency: list[set[int]] = [set() for _ in range(order)]
    for left, right in edges:
        i = int(left[1:])
        j = int(right[1:])
        adjacency[i].add(j)
        adjacency[j].add(i)
    unseen = set(range(order))
    sizes: list[int] = []
    while unseen:
        root = unseen.pop()
        queue = deque((root,))
        size = 1
        while queue:
            vertex = queue.popleft()
            for neighbor in adjacency[vertex]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
                    size += 1
        sizes.append(size)
    return tuple(sorted(sizes))


def card_component_profile(
    deck: AnonymousGraphCardMultiset,
) -> AnonymousDeckComponentProfile:
    """Return the multiplicity histogram of card component-size multisets."""
    order, rows, card_count = _admit_deck(deck)

    # The connectivity plan has been admitted before adjacency traversal begins.
    # Component sizes are graph-isomorphism invariants, so each admitted
    # representative is profiled directly; duplicate isomorphic rows accumulate
    # under the same component-size tuple without any permutation search.
    profile_counts: Counter[tuple[int, ...]] = Counter()
    for edges, multiplicity in rows:
        profile_counts[_component_orders(order, edges)] += multiplicity
    profiles = tuple(
        CardComponentSizeProfile.model_construct(
            component_orders=component_orders,
            multiplicity=multiplicity,
        )
        for component_orders, multiplicity in sorted(profile_counts.items())
    )
    return AnonymousDeckComponentProfile._from_kernel(
        card_order=order,
        card_count=card_count,
        profiles=profiles,
    )


__all__ = [
    "MAX_CARD_COMPONENT_PROFILE_OUTPUT_BYTES",
    "MAX_CARD_COMPONENT_PROFILE_WORK",
    "card_component_profile",
]
