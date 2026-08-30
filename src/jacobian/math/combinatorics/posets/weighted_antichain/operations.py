"""Maximum-weight antichain kernel."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    WeightedAntichainResult,
    _weighted_antichain_admission_error,
)

__all__ = ["compute_maximum_weight_antichain"]


def compute_maximum_weight_antichain(
    poset: FinitePoset,
    weights: tuple[CanonicalRational, ...],
) -> WeightedAntichainResult:
    """Return the exact maximum-weight antichain.

    Uses an admitted complete subset search and chooses the lexicographically
    least antichain among equal maxima.
    """
    failure = _weighted_antichain_admission_error(poset, weights)
    if failure is not None:
        code, message = failure
        raise OperationDomainValidationError(
            location=("poset", "weights"),
            code=f"poset.weighted_antichain_{code}",
            message=message,
        )
    elements = poset.elements
    n = len(elements)
    element_index = {e: i for i, e in enumerate(elements)}

    # Build comparability: comparable[i][j] = True if i < j
    comparable = [[False] * n for _ in range(n)]
    for pair in poset.strict_order_pairs:
        i, j = element_index[pair.lower], element_index[pair.upper]
        comparable[i][j] = True

    # Weights as fractions
    weight_fracs = [w.as_fraction() for w in weights]

    best_weight = Fraction(0)
    best_set: tuple[int, ...] = ()

    # Enumerate all antichains by trying all subsets
    # For efficiency, use a recursive approach
    def _search(idx: int, current: list[int], current_weight: Fraction) -> None:
        nonlocal best_weight, best_set

        if idx == n:
            candidate = tuple(current)
            candidate_labels = tuple(elements[i] for i in candidate)
            best_labels = tuple(elements[i] for i in best_set)
            if current_weight > best_weight or (
                current_weight == best_weight and candidate_labels < best_labels
            ):
                best_weight = current_weight
                best_set = candidate
            return

        # Option 1: exclude element idx
        _search(idx + 1, current, current_weight)

        # Option 2: include element idx (if compatible with current)
        can_include = True
        for j in current:
            if comparable[idx][j] or comparable[j][idx]:
                can_include = False
                break

        if can_include and weight_fracs[idx] >= 0:
            new_current = [*current, idx]
            _search(
                idx + 1,
                new_current,
                current_weight + weight_fracs[idx],
            )

    _search(0, [], Fraction(0))

    return WeightedAntichainResult(
        poset_digest=poset.poset_digest,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        maximum_antichain=tuple(elements[i] for i in best_set),
        method="EXACT_BOUNDED_SUBSET_SEARCH",
    )
