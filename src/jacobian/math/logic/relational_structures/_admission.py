"""Relational homomorphism admission shared by the native and catalog paths.

The defining invariant — for every relation symbol ``R`` and every tuple
``t`` in ``R^A``, ``h(t)`` lies in ``R^B`` — is replayed exhaustively by the
kernel, so the complete transport work ``sum |R^A|`` is preflighted here
before any tuple is transported. Structural carrier-map defects and
signature mismatches are typed domain rejections; a structurally valid
request whose transport work exceeds the published envelope is a resource
admission failure.
"""

from __future__ import annotations

from collections.abc import Sequence

from jacobian.catalog.models import (
    OperationDomainValidationError,
    OperationResourceAdmissionError,
)
from jacobian.math.logic.relational_structures.values import (
    MAX_RELATIONAL_TRANSPORT_TUPLES,
    FiniteRelationalStructure,
)

# Exhaustive homomorphism search work: every candidate carrier map is
# replayed over every source tuple. The candidate cap bounds enumeration
# overhead (map construction is linear in the source carrier); the joint
# work cap bounds tuple replays, the dominant cost.
MAX_SEARCH_CANDIDATES = 65_536
MAX_SEARCH_TUPLE_REPLAYS = 1_048_576
# An induced embedding must decide membership for every tuple in every
# Cartesian relation domain, not only the sparse positive rows.
MAX_EMBEDDING_REFLECTION_CELLS = 1_048_576


def candidate_space(source_size: int, target_size: int) -> int:
    """Return ``|B|^|A|``, the complete carrier-map count.

    The empty source admits exactly the empty map; a nonempty source
    into the empty carrier admits no map at all.
    """

    if source_size == 0:
        return 1
    if target_size == 0:
        return 0
    return int(pow(target_size, source_size))


def core_search_work(source_size: int, transport_tuples: int) -> int:
    """Return the worst-case core-iteration replay work.

    Retraction strictly shrinks the carrier, so at most one exhaustive
    endomorphism scan runs per size from ``source_size`` down to 1; the
    transport tables only shrink along restrictions, hence the source
    count bounds every level.
    """

    return int(
        sum(pow(size, size) for size in range(1, source_size + 1))
        * max(transport_tuples, 1)
    )


def admit_homomorphism_check(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> int:
    """Preflight one exhaustive homomorphism replay; return its tuple count.

    Raises ``OperationDomainValidationError`` when the two structures do not
    share one signature or the candidate map is not a total function from the
    source carrier into the target carrier, and
    ``OperationResourceAdmissionError`` when the complete transport work
    exceeds the published envelope.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    if len(carrier_map) != source.carrier_size:
        raise OperationDomainValidationError(
            location=("carrier_map",),
            code="relational.homomorphism.carrier_map_axis",
            message=(
                "a candidate homomorphism must be a total function on the "
                f"complete source carrier; expected {source.carrier_size} "
                f"images, got {len(carrier_map)}"
            ),
        )
    for position, image in enumerate(carrier_map):
        if not isinstance(image, int) or isinstance(image, bool):
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message="carrier map images must be exact integers",
            )
        if not 0 <= image < target.carrier_size:
            raise OperationDomainValidationError(
                location=("carrier_map", position),
                code="relational.homomorphism.carrier_map_value",
                message=(
                    "every carrier map image must belong to the target "
                    f"carrier 0..{target.carrier_size - 1}"
                ),
            )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    if transport_tuples > MAX_RELATIONAL_TRANSPORT_TUPLES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.transport_bound",
            message=(
                f"the exhaustive replay transports {transport_tuples} source "
                f"relation tuples, exceeding the "
                f"{MAX_RELATIONAL_TRANSPORT_TUPLES}-tuple envelope"
            ),
        )
    return transport_tuples


def embedding_reflection_cells(source: FiniteRelationalStructure) -> int:
    """Return the complete Cartesian cells replayed by an embedding."""

    return sum(
        1 if symbol.arity == 0 else source.carrier_size**symbol.arity
        for symbol in source.signature
    )


def admit_embedding_search(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> tuple[int, int, int]:
    """Admit a complete induced-embedding search.

    In addition to positive homomorphism rows, every candidate pays for the
    full source Cartesian relation domains.  This is what distinguishes an
    induced embedding from an injective homomorphism.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    space = candidate_space(source.carrier_size, target.carrier_size)
    if space > MAX_SEARCH_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_space",
            message=(
                f"the exhaustive search spans {space} carrier maps from a "
                f"{source.carrier_size}-element source into a "
                f"{target.carrier_size}-element target, exceeding the "
                f"{MAX_SEARCH_CANDIDATES}-candidate envelope"
            ),
        )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    reflection_cells = embedding_reflection_cells(source)
    target_materialization = sum(len(table) for table in target.relation_tables)
    work = target_materialization + space * (
        max(transport_tuples, 1) + reflection_cells
    )
    if reflection_cells > MAX_EMBEDDING_REFLECTION_CELLS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.embedding.reflection_bound",
            message=(
                f"induced embedding reflection inspects {reflection_cells} "
                f"Cartesian cells, exceeding the "
                f"{MAX_EMBEDDING_REFLECTION_CELLS}-cell envelope"
            ),
        )
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.embedding.search_work",
            message=(
                f"the induced embedding search materializes {target_materialization} "
                f"target rows and replays {reflection_cells} reflection cells and "
                f"{transport_tuples} positive rows across {space} maps, exceeding "
                f"the {MAX_SEARCH_TUPLE_REPLAYS}-unit envelope"
            ),
        )
    return space, transport_tuples, reflection_cells


def admit_homomorphism_search(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> tuple[int, int]:
    """Preflight one exhaustive homomorphism search; return space and tuples.

    Returns the ``(|B|^|A| carrier-map count, source tuple count)`` pair.
    Raises ``OperationDomainValidationError`` when the two structures do
    not share one signature, and ``OperationResourceAdmissionError`` when
    the candidate space or the joint replay work exceeds the published
    search envelope.
    """

    if source.signature != target.signature:
        raise OperationDomainValidationError(
            location=("target",),
            code="relational.homomorphism.signature_mismatch",
            message=(
                "source and target structures must be declared over one "
                "shared signature; signature transport is a separate "
                "explicit map, not an implicit coercion"
            ),
        )
    space = candidate_space(source.carrier_size, target.carrier_size)
    if space > MAX_SEARCH_CANDIDATES:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_space",
            message=(
                f"the exhaustive search spans {space} carrier maps from a "
                f"{source.carrier_size}-element source into a "
                f"{target.carrier_size}-element target, exceeding the "
                f"{MAX_SEARCH_CANDIDATES}-candidate envelope"
            ),
        )
    transport_tuples = sum(len(table) for table in source.relation_tables)
    work = space * max(transport_tuples, 1)
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.homomorphism.search_work",
            message=(
                f"the exhaustive search replays {transport_tuples} source "
                f"relation tuples across {space} carrier maps, exceeding the "
                f"{MAX_SEARCH_TUPLE_REPLAYS}-replay envelope"
            ),
        )
    return space, transport_tuples


def admit_core_computation(source: FiniteRelationalStructure) -> int:
    """Preflight one core iteration; return its worst-case replay work.

    Raises ``OperationResourceAdmissionError`` when the summed
    level-by-level endomorphism replay work exceeds the published
    envelope shared with homomorphism search.
    """

    transport_tuples = sum(len(table) for table in source.relation_tables)
    work = core_search_work(source.carrier_size, transport_tuples)
    if work > MAX_SEARCH_TUPLE_REPLAYS:
        raise OperationResourceAdmissionError(
            location=("source",),
            code="relational.core.search_work",
            message=(
                f"the core iteration replays {transport_tuples} source "
                f"relation tuples across shrinking endomorphism spaces, "
                f"exceeding the {MAX_SEARCH_TUPLE_REPLAYS}-replay envelope"
            ),
        )
    return work


__all__ = [
    "MAX_EMBEDDING_REFLECTION_CELLS",
    "MAX_RELATIONAL_TRANSPORT_TUPLES",
    "MAX_SEARCH_CANDIDATES",
    "MAX_SEARCH_TUPLE_REPLAYS",
    "admit_core_computation",
    "admit_embedding_search",
    "admit_homomorphism_check",
    "admit_homomorphism_search",
    "candidate_space",
    "core_search_work",
    "embedding_reflection_cells",
]
