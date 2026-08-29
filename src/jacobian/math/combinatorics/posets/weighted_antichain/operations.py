"""Maximum weight antichain kernel using exhaustive search."""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

from jacobian._exact import CanonicalRational
from jacobian.math.combinatorics.posets.core._models import FinitePoset
from jacobian.math.combinatorics.posets.weighted_antichain._models import (
    MaximumWeightAntichainResult,
)

__all__ = ["compute_maximum_weight_antichain"]


def compute_maximum_weight_antichain(
    poset: FinitePoset,
    weights: tuple[CanonicalRational, ...],
) -> MaximumWeightAntichainResult:
    """Return the exact maximum weight antichain and a witness.

    For small posets (<= 16 elements), uses exhaustive search over
    all subsets.
    """
    elements = list(poset.elements)
    n = len(elements)
    weight_fracs = [w.as_fraction() for w in weights]

    comparable = _build_comparable(poset, elements)

    best_weight = Fraction(0)
    best_antichain: tuple[str, ...] = ()

    for subset in _all_subsets(n):
        if _is_antichain(subset, comparable):
            total = sum(weight_fracs[i] for i in subset)
            if total > best_weight or (
                total == best_weight
                and _subset_to_elements(subset, elements) < best_antichain
            ):
                best_weight = total
                best_antichain = _subset_to_elements(subset, elements)

    return MaximumWeightAntichainResult(
        poset=poset,
        weights=weights,
        maximum_weight=CanonicalRational.from_fraction(best_weight),
        antichain=best_antichain,
    )


def _build_comparable(poset: FinitePoset, elements: list[str]) -> set[tuple[int, int]]:
    idx = {e: i for i, e in enumerate(elements)}
    comparable: set[tuple[int, int]] = set()
    for pair in poset.strict_order_pairs:
        i, j = idx[pair.lower], idx[pair.upper]
        comparable.add((i, j))
        comparable.add((j, i))
    return comparable


def _all_subsets(n: int):
    yield from (subset for r in range(n + 1) for subset in combinations(range(n), r))


def _is_antichain(subset: tuple[int, ...], comparable: set[tuple[int, int]]) -> bool:
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            if (subset[i], subset[j]) in comparable:
                return False
    return True


def _subset_to_elements(
    subset: tuple[int, ...], elements: list[str]
) -> tuple[str, ...]:
    return tuple(elements[i] for i in subset)
