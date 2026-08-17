"""Domain-owned additive combinatorics operations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.additive_combinatorics._models import (
    AdditiveEnergyRequest,
    AdditiveEnergyResult,
    DirectSumPredicateRequest,
    DirectSumPredicateResult,
    FiniteIntegerSet,
    RepresentationProfileEntry,
    RepresentationProfileRequest,
    RepresentationProfileResult,
    SumsetCardinalityRequest,
    SumsetCardinalityResult,
)


def _parse_set(spec: FiniteIntegerSet) -> frozenset[int]:
    return frozenset(parse_canonical_integer(element) for element in spec.elements)


def _representation_function(
    left: frozenset[int],
    right: frozenset[int],
) -> Counter[int]:
    """Return ``r_{A+B}`` as a multiset from integers to positive counts."""
    return Counter(a + b for a in left for b in right)


def _sorted_sums(counts: Mapping[int, int]) -> list[int]:
    return sorted(counts.keys())


def compute_representation_profile(
    request: RepresentationProfileRequest,
) -> RepresentationProfileResult:
    """Compute ``r_{A+B}(x)`` for every sum ``x``.

    The representation function counts, for each integer ``x``, the number of
    ordered pairs ``(a, b)`` with ``a`` in ``A`` and ``b`` in ``B`` such that
    ``a + b = x``.  Empty sets produce an empty profile.
    """
    left = _parse_set(request.left)
    right = _parse_set(request.right)
    counts = _representation_function(left, right)
    entries = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value),
            multiplicity=count,
        )
        for value in _sorted_sums(counts)
        for count in (counts[value],)
    )
    return RepresentationProfileResult(entries=entries)


def compute_additive_energy(
    request: AdditiveEnergyRequest,
) -> AdditiveEnergyResult:
    """Compute ``E(A, B) = sum_x r_{A+B}(x)^2``.

    Returns the exact additive energy together with its per-sum decomposition
    so that the squared multiplicities are individually inspectable.
    """
    left = _parse_set(request.left)
    right = _parse_set(request.right)
    counts = _representation_function(left, right)
    decomposition = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value),
            multiplicity=count,
        )
        for value in _sorted_sums(counts)
        for count in (counts[value],)
    )
    energy = sum(count * count for count in counts.values())
    return AdditiveEnergyResult(
        energy=energy,
        decomposition=decomposition,
    )


def compute_sumset_cardinality(
    request: SumsetCardinalityRequest,
) -> SumsetCardinalityResult:
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``)."""
    left = _parse_set(request.left)
    right = _parse_set(request.right)
    counts = _representation_function(left, right)
    support = tuple(format_canonical_integer(value) for value in _sorted_sums(counts))
    return SumsetCardinalityResult(
        cardinality=len(support),
        support=support,
    )


def decide_direct_sum_predicate(
    request: DirectSumPredicateRequest,
) -> DirectSumPredicateResult:
    """Decide whether ``A (\\oplus) B = Z_n`` inside the cyclic group of order ``n``.

    Every element of ``A`` and ``B`` is reduced modulo ``n``.  The predicate
    holds exactly when the map ``(a, b) -> (a + b) mod n`` is a bijection onto
    ``{0, 1, ..., n - 1}``.  This is the exact direct-factorization condition:
    every residue class admits a unique representation.  Diagnostics list
    representatives of every residue, residues with multiple representations
    (collisions), and residues with no representation (missing).
    """
    modulus = request.modulus
    if modulus < 2:
        raise ValueError("cyclic group modulus must be at least 2")

    left = {a % modulus for a in _parse_set(request.left)}
    right = {b % modulus for b in _parse_set(request.right)}

    if len(left) * len(right) != modulus:
        # |A| * |B| must equal n for a direct factorization.  Even though this
        # is a necessary condition, we still produce the full diagnostic below.
        pass

    representatives: dict[int, int] = {}
    collisions: set[int] = set()
    for a in sorted(left):
        for b in sorted(right):
            residue = (a + b) % modulus
            if residue in representatives:
                collisions.add(residue)
            representatives[residue] = residue

    full_support = set(range(modulus))
    missing = sorted(full_support - set(representatives.keys()))
    collisions_sorted = sorted(collisions)
    reps_sorted = sorted(representatives.keys())

    return DirectSumPredicateResult(
        holds=not (collisions_sorted or missing),
        modulus=modulus,
        representatives=tuple(format_canonical_integer(r) for r in reps_sorted),
        collisions=tuple(format_canonical_integer(r) for r in collisions_sorted),
        missing=tuple(format_canonical_integer(r) for r in missing),
    )


__all__ = [
    "compute_additive_energy",
    "compute_representation_profile",
    "compute_sumset_cardinality",
    "decide_direct_sum_predicate",
]
