"""Native exact kernels for additive combinatorics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from pydantic_core import PydanticCustomError

from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive import _multiset_sum
from jacobian.math.combinatorics.additive._models import (
    _MAX_RESULT_SIZE,
    AdditiveEnergyResult,
    DirectSumPredicateResult,
    FiniteIntegerSet,
    IntegerVectorSet,
    MultisetSumRepresentationProfileResult,
    MultisetSumWindow,
    OrderedDifferenceEntry,
    OrderedDifferencePair,
    OrderedDifferenceProfileResult,
    RepresentationProfileEntry,
    RepresentationProfileResult,
    SumsetCardinalityResult,
    _multiset_sum_source_values,
    _require_bounded_cartesian_product,
    _require_direct_sum_result_transport_bound,
    _vector_from_ints,
)
from jacobian.math.combinatorics.additive._multiset_sum import (
    MAX_ENUMERATION_WORK,
    MAX_SUPPORT_SIZE,
    count_sums,
)
from jacobian.math.combinatorics.additive._subset_sum_profile import (
    subset_sum_profile_counts,
    subset_sum_profile_envelope,
)
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    SubsetSumProfile,
)


def subset_sum_profile(source: IndexedIntegerSequence) -> SubsetSumProfile:
    """Return the complete indexed-subset multiplicity profile of ``source``.

    Every position is independently selected at most once. Equal values and
    zeros therefore contribute separate multiplicity even though they may not
    enlarge the numeric support. The empty subset is included by definition.
    """

    envelope = subset_sum_profile_envelope(source)
    counts = subset_sum_profile_counts(source)
    if len(counts) > envelope.support_bound:
        raise RuntimeError("subset-sum support exceeded its admitted bound")
    return SubsetSumProfile._from_kernel(source, counts)


def _parse_set(spec: FiniteIntegerSet) -> frozenset[int]:
    return frozenset(parse_canonical_integer(element) for element in spec.elements)


def _representation_function(
    left: frozenset[int], right: frozenset[int]
) -> Counter[int]:
    """Return ``r_{A+B}`` as a multiset from integers to positive counts."""
    return Counter(a + b for a in left for b in right)


def _sorted_sums(counts: Mapping[int, int]) -> list[int]:
    return sorted(counts.keys())


def _admit_direct_sum(
    modulus: int, left: FiniteIntegerSet, right: FiniteIntegerSet
) -> None:
    if type(modulus) is not int or not 2 <= modulus <= _MAX_RESULT_SIZE:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="additive_combinatorics.direct_sum.modulus_domain",
            message=(
                "direct-sum modulus must be an integer between 2 and "
                f"{_MAX_RESULT_SIZE:,}"
            ),
        )
    try:
        _require_bounded_cartesian_product(left, right)
        _require_direct_sum_result_transport_bound(modulus, left, right)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("left", "right"), code=exc.type, message=exc.message()
        ) from None


def representation_profile(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> RepresentationProfileResult:
    """Compute ``r_{A+B}(x)`` for every sum ``x``."""
    _require_bounded_cartesian_product(left, right)
    counts = _representation_function(_parse_set(left), _parse_set(right))
    entries = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value), multiplicity=counts[value]
        )
        for value in _sorted_sums(counts)
    )
    return RepresentationProfileResult(entries=entries)


def _admit_multiset_sum_profile(
    values: tuple[int, ...], arity: int, window: MultisetSumWindow | None
) -> None:
    """Admit exact multiset enumeration and output support before the kernel."""
    candidate_count = _multiset_sum.candidate_count(len(values), arity)
    bounds = window.as_integer_bounds() if window is not None else None
    work = _multiset_sum.enumeration_work(values, arity, bounds, candidate_count)
    if work > MAX_ENUMERATION_WORK:
        raise OperationDomainValidationError(
            location=("arity",),
            code="additive_combinatorics.multiset_sum.work_bound",
            message=(
                f"multiset-sum enumeration requires {work} coordinate steps, "
                f"exceeding the {MAX_ENUMERATION_WORK}-step bound"
            ),
        )
    support_bound = _multiset_sum.support_bound(values, arity, bounds, candidate_count)
    if support_bound > MAX_SUPPORT_SIZE:
        raise OperationDomainValidationError(
            location=("arity",),
            code="additive_combinatorics.multiset_sum.support_bound",
            message=(
                f"multiset-sum profile may contain {support_bound} rows, exceeding "
                f"the {MAX_SUPPORT_SIZE}-row result bound; supply a "
                "narrower closed sum window"
            ),
        )


def multiset_sum_representation_profile(
    source: FiniteIntegerSet,
    arity: int,
    window: MultisetSumWindow | None = None,
) -> MultisetSumRepresentationProfileResult:
    """Count fixed-arity unordered source multisets by their exact sum."""
    try:
        values = _multiset_sum_source_values(source)
    except ValueError as exc:
        raise OperationDomainValidationError(
            location=("source",),
            code="additive_combinatorics.multiset_sum.source_domain",
            message=str(exc),
        ) from exc
    _admit_multiset_sum_profile(values, arity, window)
    bounds = window.as_integer_bounds() if window is not None else None
    counts = count_sums(values, arity, bounds)
    entries = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value), multiplicity=counts[value]
        )
        for value in sorted(counts)
    )
    return MultisetSumRepresentationProfileResult._from_kernel(
        source, arity, window, entries
    )


def additive_energy(
    left: FiniteIntegerSet, right: FiniteIntegerSet
) -> AdditiveEnergyResult:
    """Compute ``E(A, B) = sum_x r_{A+B}(x)^2``."""
    _require_bounded_cartesian_product(left, right)
    counts = _representation_function(_parse_set(left), _parse_set(right))
    decomposition = tuple(
        RepresentationProfileEntry(
            sum=format_canonical_integer(value), multiplicity=counts[value]
        )
        for value in _sorted_sums(counts)
    )
    return AdditiveEnergyResult._from_kernel(
        sum(count * count for count in counts.values()), decomposition
    )


def sumset_cardinality(
    left: FiniteIntegerSet, right: FiniteIntegerSet
) -> SumsetCardinalityResult:
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``)."""
    _require_bounded_cartesian_product(left, right)
    counts = _representation_function(_parse_set(left), _parse_set(right))
    support = tuple(format_canonical_integer(value) for value in _sorted_sums(counts))
    return SumsetCardinalityResult._from_kernel(support)


def direct_sum_predicate(
    modulus: int, left: FiniteIntegerSet, right: FiniteIntegerSet
) -> DirectSumPredicateResult:
    """Decide whether ``A (\\oplus) B = Z_n`` inside the cyclic group."""
    _admit_direct_sum(modulus, left, right)
    left_values = {a % modulus for a in _parse_set(left)}
    right_values = {b % modulus for b in _parse_set(right)}
    representatives: dict[int, int] = {}
    collisions: set[int] = set()
    for left_value in sorted(left_values):
        for right_value in sorted(right_values):
            residue = (left_value + right_value) % modulus
            if residue in representatives:
                collisions.add(residue)
            representatives[residue] = residue
    missing = sorted(set(range(modulus)) - set(representatives))
    collisions_sorted = sorted(collisions)
    representatives_sorted = sorted(representatives)
    return DirectSumPredicateResult._from_kernel(
        holds=not (collisions_sorted or missing),
        modulus=modulus,
        representatives=tuple(
            format_canonical_integer(value) for value in representatives_sorted
        ),
        collisions=tuple(
            format_canonical_integer(value) for value in collisions_sorted
        ),
        missing=tuple(format_canonical_integer(value) for value in missing),
    )


def ordered_difference_profile(
    vectors: IntegerVectorSet,
) -> OrderedDifferenceProfileResult:
    """Compute the complete ordered-difference profile of a finite vector set."""
    vector_values = [vector.as_int_tuple() for vector in vectors.vectors]
    dimension = len(vector_values[0])
    difference_map: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for left_index, left in enumerate(vector_values):
        for right_index, right in enumerate(vector_values):
            if left_index == right_index:
                continue
            difference = tuple(left[index] - right[index] for index in range(dimension))
            if difference == (0,) * dimension:
                continue
            difference_map.setdefault(difference, []).append((left_index, right_index))
    entries: list[OrderedDifferenceEntry] = []
    total_pairs = 0
    maximum = 0
    first_collision = None
    for difference in sorted(difference_map):
        pairs = difference_map[difference]
        multiplicity = len(pairs)
        total_pairs += multiplicity
        maximum = max(maximum, multiplicity)
        entries.append(
            OrderedDifferenceEntry(
                difference=_vector_from_ints(difference),
                multiplicity=multiplicity,
                pairs=tuple(
                    OrderedDifferencePair(left_index=left, right_index=right)
                    for left, right in pairs
                ),
            )
        )
    if maximum > 1:
        first_collision = next(
            entry.pairs[0] for entry in entries if entry.multiplicity > 1
        )
    return OrderedDifferenceProfileResult._from_kernel(
        vectors,
        dimension=dimension,
        total_ordered_pairs=total_pairs,
        support_size=len(entries),
        max_multiplicity=maximum,
        entries=tuple(entries),
        has_repeated_difference=maximum > 1,
        first_collision=first_collision,
    )


__all__ = [
    "additive_energy",
    "direct_sum_predicate",
    "multiset_sum_representation_profile",
    "ordered_difference_profile",
    "representation_profile",
    "subset_sum_profile",
    "sumset_cardinality",
]
