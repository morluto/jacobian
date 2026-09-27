"""Exact bounded equality for anonymous graph-card multisets."""

from math import comb

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.graphs.decks._models import (
    MAX_ANONYMOUS_CARD_CLASSES,
    MAX_UNLABELLED_DECK_VERTICES,
    AnonymousGraphCardClass,
    AnonymousGraphCardMultiset,
    _anonymous_canonicalization_work,
    _canonical_card_edges,
)
from jacobian.math.graphs.decks.anonymous_equality._models import (
    AnonymousDeckEqualityResult,
)
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_ANONYMOUS_DECK_EQUALITY_WORK = 2_000_000
_CanonicalRow = tuple[tuple[tuple[str, str], ...], int]


def _admit_multiset(
    value: object, name: str
) -> tuple[int, tuple[_CanonicalRow, ...], int]:
    if type(value) is not AnonymousGraphCardMultiset:
        raise OperationDomainValidationError(
            location=(name,),
            code="graph_deck.equality_carrier",
            message="both inputs must be AnonymousGraphCardMultiset values",
        )
    order = getattr(value, "card_order", None)
    classes = getattr(value, "classes", None)
    if (
        type(order) is not int
        or order < 0
        or order > MAX_UNLABELLED_DECK_VERTICES
        or type(classes) is not tuple
        or len(classes) > MAX_ANONYMOUS_CARD_CLASSES
    ):
        raise OperationDomainValidationError(
            location=(name,),
            code="graph_deck.equality_shape",
            message="anonymous card multiset has a malformed bounded shape",
        )
    # Check all fields before canonicalization. In particular, reject forged
    # model_construct values before using labels as dictionary keys.
    expected_vertices = tuple(f"v{i:02d}" for i in range(order))
    pair_count = comb(order, 2)
    checked: list[_CanonicalRow] = []
    for index, item in enumerate(classes):
        if type(item) is not AnonymousGraphCardClass:
            raise OperationDomainValidationError(
                location=(name, "classes", index),
                code="graph_deck.equality_class_carrier",
                message="each class must use AnonymousGraphCardClass",
            )
        representative = getattr(item, "representative", None)
        multiplicity = getattr(item, "multiplicity", None)
        # Read the representative's fields with getattr: a forged model_construct
        # value may be the exact carrier type while omitting ``vertices`` or
        # ``edges``, and direct reads would raise AttributeError instead of the
        # operation's typed domain error.
        representative_vertices = getattr(representative, "vertices", None)
        representative_edges = getattr(representative, "edges", None)
        if (
            type(representative) is not SimpleUndirectedGraph
            or type(representative_vertices) is not tuple
            or representative_vertices != expected_vertices
            or type(representative_edges) is not tuple
            or len(representative_edges) > pair_count
            or type(multiplicity) is not int
            or not 1 <= multiplicity < 10**12
        ):
            raise OperationDomainValidationError(
                location=(name, "classes", index),
                code="graph_deck.equality_class",
                message="each class must have a bounded representative and positive multiplicity",
            )
        for edge in representative_edges:
            if (
                type(edge) is not tuple
                or len(edge) != 2
                or any(type(label) is not str for label in edge)
                or edge[0] >= edge[1]
                or edge[0] not in expected_vertices
                or edge[1] not in expected_vertices
            ):
                raise OperationDomainValidationError(
                    location=(name, "classes", index, "representative", "edges"),
                    code="graph_deck.equality_edges",
                    message="representative edges must be canonical pairs on the fixed card axis",
                )
        if tuple(sorted(set(representative_edges))) != representative_edges:
            raise OperationDomainValidationError(
                location=(name, "classes", index, "representative", "edges"),
                code="graph_deck.equality_edges",
                message="representative edges must be unique and ordered",
            )
        checked.append((representative_edges, multiplicity))

    return order, tuple(checked), _anonymous_canonicalization_work(order, len(classes))


def _canonical_multiplicities(
    order: int, checked: tuple[_CanonicalRow, ...]
) -> dict[tuple[tuple[str, str], ...], int]:
    expected_vertices = tuple(f"v{i:02d}" for i in range(order))
    result: dict[tuple[tuple[str, str], ...], int] = {}
    for edges, multiplicity in checked:
        canonical = _canonical_card_edges(expected_vertices, edges)
        result[canonical] = result.get(canonical, 0) + multiplicity
    return result


def anonymous_deck_equality(
    left: object,
    right: object,
) -> AnonymousDeckEqualityResult:
    """Compare class multiplicities, independent of each card's vertex labels."""
    left_order, left_checked, left_work = _admit_multiset(left, "left")
    right_order, right_checked, right_work = _admit_multiset(right, "right")
    if left_work + right_work > MAX_ANONYMOUS_DECK_EQUALITY_WORK:
        raise OperationResourceAdmissionError(
            location=("inputs",),
            code="graph_deck.equality_work_bound",
            message="deck equality exceeds the exact aggregate canonicalization work bound",
        )
    left_counts = _canonical_multiplicities(left_order, left_checked)
    right_counts = _canonical_multiplicities(right_order, right_checked)
    return AnonymousDeckEqualityResult(
        equal=left_order == right_order and left_counts == right_counts
    )


__all__ = ["MAX_ANONYMOUS_DECK_EQUALITY_WORK", "anonymous_deck_equality"]
