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

    # Extend only valid prefixes.  This avoids revisiting every pair inside
    # every candidate combination while retaining cardinality/lexicographic
    # output order.
    by_size: list[list[tuple[int, tuple[int, ...]]]] = [
        [] for _ in range(max_cardinality + 1)
    ]
    by_size[0].append((0, ()))
    for size in range(1, max_cardinality + 1):
        for mask, prefix in by_size[size - 1]:
            start = prefix[-1] + 1 if prefix else 0
            for index in range(start, n):
                if not comparable[index] & mask:
                    by_size[size].append((mask | (1 << index), (*prefix, index)))
        if size >= max(1, min_cardinality):
            antichains.extend(
                tuple(elements[index] for index in prefix)
                for _mask, prefix in by_size[size]
            )

    return AntichainEnumerationResult(
        poset_digest=poset.poset_digest,
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        antichains=tuple(antichains),
        count=len(antichains),
    )
