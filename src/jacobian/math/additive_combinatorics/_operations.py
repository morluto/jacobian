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
    MultisetSumRepresentationProfileRequest,
    MultisetSumRepresentationProfileResult,
    OrderedDifferenceEntry,
    OrderedDifferencePair,
    OrderedDifferenceProfileRequest,
    OrderedDifferenceProfileResult,
    RepresentationProfileEntry,
    RepresentationProfileRequest,
    RepresentationProfileResult,
    SubsetSumProfileRequest,
    SumsetCardinalityRequest,
    SumsetCardinalityResult,
    _multiset_sum_source_values,
    _vector_from_ints,
)
from jacobian.math.additive_combinatorics._multiset_sum import count_sums
from jacobian.math.additive_combinatorics._subset_sum_profile import (
    subset_sum_profile_counts,
    subset_sum_profile_envelope,
)
from jacobian.math.additive_combinatorics.operations import subset_sum_profile
from jacobian.math.additive_combinatorics.values import SubsetSumProfile


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


def compute_multiset_sum_representation_profile(
    request: MultisetSumRepresentationProfileRequest,
) -> MultisetSumRepresentationProfileResult:
    """Count fixed-arity unordered source multisets by their exact sum.

    The source's numeric order is the index order. Every nondecreasing index
    tuple of the requested arity is counted once, including repeated indices.
    An optional closed window filters sums but not candidate inspection, so the
    returned rows remain complete for that exact mathematical scope.
    """
    values = _multiset_sum_source_values(request.source)
    bounds = request.window.as_integer_bounds() if request.window is not None else None
    counts = count_sums(values, request.arity, bounds)
    entries = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value),
            multiplicity=counts[value],
        )
        for value in sorted(counts)
    )
    return MultisetSumRepresentationProfileResult._from_kernel(request, entries)


def compute_subset_sum_profile(
    request: SubsetSumProfileRequest,
) -> SubsetSumProfile:
    """Compute the complete exact indexed-subset sum profile."""

    return subset_sum_profile(request.source)


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
    return AdditiveEnergyResult._from_kernel(energy, decomposition)


def compute_sumset_cardinality(
    request: SumsetCardinalityRequest,
) -> SumsetCardinalityResult:
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``)."""
    left = _parse_set(request.left)
    right = _parse_set(request.right)
    counts = _representation_function(left, right)
    support = tuple(format_canonical_integer(value) for value in _sorted_sums(counts))
    return SumsetCardinalityResult._from_kernel(support)


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

    return DirectSumPredicateResult._from_kernel(
        holds=not (collisions_sorted or missing),
        modulus=modulus,
        representatives=tuple(format_canonical_integer(r) for r in reps_sorted),
        collisions=tuple(format_canonical_integer(r) for r in collisions_sorted),
        missing=tuple(format_canonical_integer(r) for r in missing),
    )


__all__ = [
    "compute_additive_energy",
    "compute_multiset_sum_representation_profile",
    "compute_ordered_difference_profile",
    "compute_representation_profile",
    "compute_subset_sum_profile",
    "compute_sumset_cardinality",
    "decide_direct_sum_predicate",
]


def compute_ordered_difference_profile(
    request: OrderedDifferenceProfileRequest,
) -> OrderedDifferenceProfileResult:
    """Compute r_{A-A}(v) for every nonzero difference vector v.

    For a finite set A in Z^d, the ordered-difference profile counts,
    for every nonzero difference v, the number of ordered pairs (x, y)
    in A^2 with x != y and x - y = v. Each entry includes the source
    pairs so collision classes are directly inspectable.
    """
    vectors = [vec.as_int_tuple() for vec in request.vectors.vectors]
    n = len(vectors)
    dimension = len(vectors[0])

    difference_map: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = tuple(vectors[i][k] - vectors[j][k] for k in range(dimension))
            if diff == tuple(0 for _ in range(dimension)):
                continue
            difference_map.setdefault(diff, []).append((i, j))

    entries: list[OrderedDifferenceEntry] = []
    total_pairs = 0
    max_mult = 0
    first_collision = None

    for diff in sorted(difference_map):
        pairs = difference_map[diff]
        multiplicity = len(pairs)
        total_pairs += multiplicity
        max_mult = max(max_mult, multiplicity)
        pair_models = tuple(
            OrderedDifferencePair(left_index=i, right_index=j) for i, j in pairs
        )
        entries.append(
            OrderedDifferenceEntry(
                difference=_vector_from_ints(diff),
                multiplicity=multiplicity,
                pairs=pair_models,
            )
        )

    has_repeated = max_mult > 1
    if has_repeated:
        for entry in entries:
            if entry.multiplicity > 1:
                first_collision = entry.pairs[0]
                break

    return OrderedDifferenceProfileResult._from_kernel(
        request,
        dimension=dimension,
        total_ordered_pairs=total_pairs,
        support_size=len(entries),
        max_multiplicity=max_mult,
        entries=tuple(entries),
        has_repeated_difference=has_repeated,
        first_collision=first_collision,
    )


def verify_subset_sum_profile(profile: SubsetSumProfile) -> bool:
    """Check an independently supplied complete profile within its envelope."""

    envelope = subset_sum_profile_envelope(profile.source)
    expected = tuple(sorted(subset_sum_profile_counts(profile.source).items()))
    actual = tuple(
        (
            parse_canonical_integer(entry.sum),
            parse_canonical_integer(entry.multiplicity),
        )
        for entry in profile.entries
    )
    return (
        len(expected) <= envelope.support_bound
        and actual == expected
        and parse_canonical_integer(profile.total_subsets)
        == 1 << len(profile.source.items)
    )


def verify_multiset_sum_representation_profile(
    result: MultisetSumRepresentationProfileResult,
) -> bool:
    """Check a supplied complete multiset-sum profile under request admission."""

    request = MultisetSumRepresentationProfileRequest(
        source=result.source, arity=result.arity, window=result.window
    )
    values = _multiset_sum_source_values(request.source)
    bounds = request.window.as_integer_bounds() if request.window is not None else None
    counts = count_sums(values, request.arity, bounds)
    expected = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value), multiplicity=counts[value]
        )
        for value in sorted(counts)
    )
    return result.entries == expected


def verify_ordered_difference_profile(result: OrderedDifferenceProfileResult) -> bool:
    """Check an independently supplied complete ordered-difference profile."""

    expected = compute_ordered_difference_profile(
        OrderedDifferenceProfileRequest(vectors=result.vectors)
    )
    return result == expected
