"""Antichain enumeration kernel."""

from __future__ import annotations

from itertools import combinations

from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.antichain_enumeration._models import (
    AntichainEnumerationResult,
    require_antichain_enumeration_envelope,
)
from jacobian.math.combinatorics.posets.core._models import FinitePoset

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
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("min_cardinality", "max_cardinality"),
            code="poset.antichain_enumeration_envelope_exceeded",
            message=str(exc),
        ) from exc
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

    for size in range(max(1, min_cardinality), max_cardinality + 1):
        if size > n:
            break
        for combo in combinations(range(n), size):
            # Check if all pairs are incomparable
            is_antichain = True
            for idx in range(len(combo)):
                for jdx in range(idx + 1, len(combo)):
                    if comparable[combo[idx]] & (1 << combo[jdx]):
                        is_antichain = False
                        break
                if not is_antichain:
                    break
            if is_antichain:
                antichains.append(tuple(elements[i] for i in combo))

    return AntichainEnumerationResult(
        poset_digest=poset.poset_digest,
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        antichains=tuple(antichains),
        count=len(antichains),
    )
