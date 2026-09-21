"""Exact bounded native kernel for relational homomorphism checking."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import lcm

from jacobian._execution import request_checkpoint
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.logic.relational_structures._admission import (
    admit_core_computation,
    admit_homomorphism_check,
    admit_homomorphism_search,
)
from jacobian.math.logic.relational_structures._models import (
    EmbeddingSearchResult,
    HomomorphismCheckResult,
    HomomorphismCoreResult,
    HomomorphismCountResult,
    HomomorphismSearchResult,
    HomomorphismSearchStatus,
    HomomorphismStatus,
    HomomorphismViolationWitness,
    SymbolTransportProfile,
)
from jacobian.math.logic.relational_structures.values import (
    FiniteRelationalStructure,
)


def check_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    carrier_map: Sequence[int],
) -> HomomorphismCheckResult:
    """Decide one candidate carrier map by exhaustive invariant replay.

    For every relation symbol ``R`` and every tuple ``t`` in the complete
    source table ``R^A``, the coordinatewise image ``h(t)`` is tested for
    membership in the complete target table ``R^B``. The replay is exhaustive
    (never sampled or short-circuited), records complete per-symbol transport
    counts, and retains the first violating ``(symbol, source tuple, image
    tuple)`` witness in deterministic signature/row order. The identity map on
    any structure is a homomorphism, and the composition of two checked
    homomorphisms is again checked as a homomorphism by this same replay.
    """

    admit_homomorphism_check(source, target, carrier_map)
    checked_map = tuple(carrier_map)
    target_tables = tuple(set(table) for table in target.relation_tables)

    witness: HomomorphismViolationWitness | None = None
    profiles: list[SymbolTransportProfile] = []
    for symbol_index, symbol in enumerate(source.signature):
        source_table = source.relation_tables[symbol_index]
        target_table = target_tables[symbol_index]
        preserved = 0
        for source_tuple in source_table:
            image_tuple = tuple(checked_map[coordinate] for coordinate in source_tuple)
            if image_tuple in target_table:
                preserved += 1
            elif witness is None:
                witness = HomomorphismViolationWitness(
                    symbol_id=symbol.symbol_id,
                    arity=symbol.arity,
                    source_tuple=source_tuple,
                    image_tuple=image_tuple,
                )
        profiles.append(
            SymbolTransportProfile(
                symbol_id=symbol.symbol_id,
                arity=symbol.arity,
                source_tuples=len(source_table),
                preserved_tuples=preserved,
            )
        )

    status = (
        HomomorphismStatus.NOT_HOMOMORPHISM
        if witness is not None
        else HomomorphismStatus.HOMOMORPHISM
    )
    return HomomorphismCheckResult._from_kernel(
        status=status,
        source=source,
        target=target,
        carrier_map=checked_map,
        witness=witness,
        symbol_profiles=tuple(profiles),
    )


def search_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> HomomorphismSearchResult:
    """Search two structures for a homomorphism by exhaustive replay.

    Candidate carrier maps run in lexicographic order (source label 0
    varying slowest) through the complete ``|B|^|A|`` space admitted up
    front. Every candidate is decided by the reused homomorphism replay,
    so a FOUND map carries its complete check and an EXHAUSTED scan is
    a proved negative: every map was examined and none transports.
    """

    if not isinstance(source, FiniteRelationalStructure) or not isinstance(
        target, FiniteRelationalStructure
    ):
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.homomorphism.structure_type",
            message="homomorphism search consumes finite relational structures",
        )
    total_candidates, _transport_tuples = admit_homomorphism_search(source, target)
    found = _first_homomorphism(
        source, target, total_candidates, require_injective=False
    )
    if found is not None:
        check, examined = found
        return HomomorphismSearchResult._from_kernel(
            status=HomomorphismSearchStatus.FOUND,
            source=source,
            target=target,
            check=check,
            candidates_examined=examined,
            total_candidates=total_candidates,
        )
    return HomomorphismSearchResult._from_kernel(
        status=HomomorphismSearchStatus.EXHAUSTED,
        source=source,
        target=target,
        check=None,
        candidates_examined=total_candidates,
        total_candidates=total_candidates,
    )


def _first_homomorphism(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
    total_candidates: int,
    *,
    require_injective: bool,
) -> tuple[HomomorphismCheckResult, int] | None:
    """Scan carrier maps in lexicographic order for the first replay that
    transports, optionally requiring distinct images.

    Returns the winning check with its one-based examination count, or
    ``None`` after a complete scan. Shared by homomorphism and embedding
    search so both see identical order and receipts.
    """

    examined = 0
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during homomorphism search enumeration")
        if require_injective and len(set(candidate)) != source.carrier_size:
            continue
        check = check_homomorphism(source, target, candidate)
        if check.status is HomomorphismStatus.HOMOMORPHISM:
            return check, examined
    if examined != total_candidates:
        raise RuntimeError("homomorphism search did not scan its admitted space")
    return None


def search_embedding(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> EmbeddingSearchResult:
    """Search two structures for an injective homomorphism.

    The same admitted space the homomorphism search scans is replayed
    in the same order, but only maps with distinct images are decided:
    FOUND retains the first embedding with its complete check and
    EXHAUSTED proves no injective map transports.
    """

    if not isinstance(source, FiniteRelationalStructure) or not isinstance(
        target, FiniteRelationalStructure
    ):
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.homomorphism.structure_type",
            message="embedding search consumes finite relational structures",
        )
    total_candidates, _transport_tuples = admit_homomorphism_search(source, target)
    found = _first_homomorphism(
        source, target, total_candidates, require_injective=True
    )
    if found is not None:
        check, examined = found
        return EmbeddingSearchResult._from_kernel(
            status=HomomorphismSearchStatus.FOUND,
            source=source,
            target=target,
            check=check,
            candidates_examined=examined,
            total_candidates=total_candidates,
        )
    return EmbeddingSearchResult._from_kernel(
        status=HomomorphismSearchStatus.EXHAUSTED,
        source=source,
        target=target,
        check=None,
        candidates_examined=total_candidates,
        total_candidates=total_candidates,
    )


def count_homomorphisms(
    source: FiniteRelationalStructure,
    target: FiniteRelationalStructure,
) -> HomomorphismCountResult:
    """Count every homomorphism by exhaustive replay without early stopping.

    The same admitted space the search scans is replayed completely:
    every carrier map is decided by the reused homomorphism replay and
    transporting maps are counted. The count is exact and complete.
    """

    if not isinstance(source, FiniteRelationalStructure) or not isinstance(
        target, FiniteRelationalStructure
    ):
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.homomorphism.structure_type",
            message="homomorphism counting consumes finite relational structures",
        )
    total_candidates, _transport_tuples = admit_homomorphism_search(source, target)
    count = 0
    examined = 0
    for candidate in product(range(target.carrier_size), repeat=source.carrier_size):
        examined += 1
        if examined % 4_096 == 0:
            request_checkpoint("during homomorphism count enumeration")
        if check_homomorphism(source, target, candidate).status is (
            HomomorphismStatus.HOMOMORPHISM
        ):
            count += 1
    if examined != total_candidates:
        raise RuntimeError("homomorphism count did not scan its admitted space")
    return HomomorphismCountResult._from_kernel(
        source=source,
        target=target,
        count=count,
        total_candidates=total_candidates,
    )


def _induced_substructure(
    structure: FiniteRelationalStructure, image: tuple[int, ...]
) -> FiniteRelationalStructure:
    """Restrict to the sorted image labels with canonical relabeling.

    Core label ``c`` denotes image label ``image[c]``; a table tuple
    survives exactly when every coordinate lies in the image, relabeled
    by rank. Tables stay canonical: filtering and relabeling preserve
    strictly increasing unique rows.
    """

    rank = {label: position for position, label in enumerate(image)}
    tables = tuple(
        tuple(
            tuple(rank[coordinate] for coordinate in row)
            for row in table
            if all(coordinate in rank for coordinate in row)
        )
        for table in structure.relation_tables
    )
    return FiniteRelationalStructure(
        carrier_size=len(image),
        signature=structure.signature,
        relation_tables=tables,
    )


def _idempotent_retraction(
    composed: tuple[int, ...],
    inclusion: tuple[int, ...],
    source_size: int,
) -> tuple[int, ...]:
    """Return the core-label retraction induced by an endomorphism onto the core.

    ``composed`` is an endomorphism of the source whose image is exactly the
    strictly increasing ``inclusion`` image. On that image it is a bijection,
    so a power of it acts as the identity there while keeping the same image;
    that power is an idempotent endomorphism, and reading its values as core
    labels gives the retraction with ``retraction[inclusion[c]] == c``.
    """

    if not inclusion:
        return ()
    position = {label: index for index, label in enumerate(inclusion)}
    permutation = tuple(position[composed[label]] for label in inclusion)
    seen = [False] * len(inclusion)
    order = 1
    for start in range(len(inclusion)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = permutation[node]
            length += 1
        order = lcm(order, length)

    def compose(first: tuple[int, ...], second: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(first[second[label]] for label in range(source_size))

    power = tuple(range(source_size))
    base = composed
    exponent = order
    while exponent:
        if exponent & 1:
            power = compose(power, base)
        base = compose(base, base)
        exponent >>= 1
    return tuple(position[power[label]] for label in range(source_size))


def compute_core(
    source: FiniteRelationalStructure,
) -> HomomorphismCoreResult:
    """Compute the minimal retract with its witnessing maps.

    Each level scans endomorphisms in lexicographic order; the first
    map with a smaller image restricts the structure to its sorted
    image with canonical relabeling, and the scan repeats. The final
    level examines every endomorphism and finds no smaller image, so
    the retained structure is minimal. The composed retraction is
    replayed once through the reused check before construction.
    """

    if not isinstance(source, FiniteRelationalStructure):
        raise OperationDomainValidationError(
            location=("source",),
            code="relational.homomorphism.structure_type",
            message="core computation consumes a finite relational structure",
        )
    admit_core_computation(source)
    current = source
    inclusion = tuple(range(source.carrier_size))
    retraction = tuple(range(source.carrier_size))
    scanned = 0
    while True:
        size = current.carrier_size
        step: tuple[int, ...] | None = None
        for candidate in product(range(size), repeat=size):
            scanned += 1
            if scanned % 4_096 == 0:
                request_checkpoint("during core endomorphism enumeration")
            check = check_homomorphism(current, current, candidate)
            if (
                check.status is HomomorphismStatus.HOMOMORPHISM
                and len(set(candidate)) < size
            ):
                step = tuple(candidate)
                break
        if step is None:
            break
        image = tuple(sorted(set(step)))
        relabel = {label: position for position, label in enumerate(image)}
        retraction = tuple(relabel[step[label]] for label in retraction)
        inclusion = tuple(inclusion[label] for label in image)
        current = _induced_substructure(current, image)
    composed = tuple(inclusion[label] for label in retraction)
    final = check_homomorphism(source, source, composed)
    if final.status is not HomomorphismStatus.HOMOMORPHISM:
        raise RuntimeError("composed core retraction is not an endomorphism")
    if tuple(sorted(set(composed))) != inclusion:
        raise RuntimeError("composed core retraction has the wrong image")
    retraction = _idempotent_retraction(composed, inclusion, source.carrier_size)
    if not all(
        retraction[inclusion[core_label]] == core_label
        for core_label in range(len(inclusion))
    ):
        raise RuntimeError("core retraction does not split its inclusion")
    return HomomorphismCoreResult._from_kernel(
        source=source,
        core=current,
        inclusion=inclusion,
        retraction=retraction,
    )


__all__ = [
    "check_homomorphism",
    "compute_core",
    "count_homomorphisms",
    "search_embedding",
    "search_homomorphism",
]
