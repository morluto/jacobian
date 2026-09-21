"""Antichain enumeration kernel."""

from __future__ import annotations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.antichain_enumeration._models import (
    AntichainEnumerationResult,
    require_antichain_enumeration_envelope,
)
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.core.operations import _admit_canonical_poset

__all__ = ["enumerate_antichains"]


def enumerate_antichains(
    poset: FinitePoset,
    min_cardinality: int,
    max_cardinality: int,
) -> AntichainEnumerationResult:
    """Return every antichain of sizes in [*min_cardinality*, *max_cardinality*].

    An antichain is a set of pairwise incomparable elements. Uses a bitset
    comparability lookup for efficient rejection.
    """
    try:
        require_antichain_enumeration_envelope(poset, min_cardinality, max_cardinality)
    except (AttributeError, TypeError):
        _admit_canonical_poset(poset)
    except ValueError as exc:
        if "finite poset carrier" in str(exc):
            _admit_canonical_poset(poset)
        else:
            raise OperationDomainValidationError(
                location=("min_cardinality", "max_cardinality"),
                code="poset.antichain_enumeration_envelope_exceeded",
                message=str(exc),
            ) from exc
    _admit_canonical_poset(poset)
    elements = poset.elements
    n = len(elements)
    element_index = {e: i for i, e in enumerate(elements)}

    # Build comparability bitset
    comparable = [0] * n
    for pair in poset.strict_order_pairs:
        lower, upper = pair.lower, pair.upper
        i, j = element_index[lower], element_index[upper]
        comparable[i] |= 1 << j
        comparable[j] |= 1 << i

    antichains: list[tuple[str, ...]] = []

    # Include empty antichain only if min_cardinality == 0
    if min_cardinality == 0:
        antichains.append(())

    # Enumerate only requested cardinality levels. Recursive extension keeps
    # one prefix per depth instead of retaining every smaller antichain. The
    # remaining-slot guard also prevents exploring prefixes that cannot reach
    # the requested size.
    def extend(
        target_size: int,
        start: int,
        mask: int,
        prefix: tuple[int, ...],
    ) -> None:
        if len(prefix) == target_size:
            antichains.append(tuple(elements[index] for index in prefix))
            return
        needed = target_size - len(prefix)
        for index in range(start, n - needed + 1):
            if not comparable[index] & mask:
                extend(
                    target_size,
                    index + 1,
                    mask | (1 << index),
                    (*prefix, index),
                )

    for size in range(max(1, min_cardinality), min(max_cardinality, n) + 1):
        extend(size, 0, 0, ())

    return AntichainEnumerationResult(
        poset_digest=poset.poset_digest,
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        antichains=tuple(antichains),
        count=len(antichains),
    )
