"""Native exact kernels for additive combinatorics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from jacobian.canonical import (
    CanonicalLimits,
)
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive import _multiset_sum
from jacobian.math.combinatorics.additive._models import (
    _MAX_COORDINATE_DIGITS,
    _MAX_DIMENSION,
    _MAX_RESULT_SIZE,
    _MAX_VECTOR_COORDINATE_LENGTH,
    _MAX_VECTOR_SET_SIZE,
    AdditiveEnergyResult,
    DirectSumPredicateResult,
    FiniteIntegerSet,
    IntegerVector,
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

MAX_DIRECT_SUM_DIAGNOSTIC_ENTRIES = 1_048_576

# The profile kernel performs one exact subtraction per source-pair coordinate.
# Charge the complete result carrier as well: source coordinates, difference
# coordinates, and the two indices retained for every possible witness pair.
MAX_ORDERED_DIFFERENCE_COORDINATE_WORK = 1_000_000
MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS = 2_000_000


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
    if any(
        abs(element) >= 10 ** CanonicalLimits().max_integer_digits
        for element in spec.elements
    ):
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="additive_combinatorics.integer_digit_bound",
            message="finite-set operands exceed the canonical integer digit bound",
        )
    return frozenset(spec.elements)


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
    if modulus > MAX_DIRECT_SUM_DIAGNOSTIC_ENTRIES:
        raise OperationDomainValidationError(
            location=("modulus",),
            code="additive_combinatorics.direct_sum.result_cardinality_exceeded",
            message=(
                "direct-sum diagnostics exceed the "
                f"{MAX_DIRECT_SUM_DIAGNOSTIC_ENTRIES:,}-entry result bound"
            ),
        )
    try:
        _require_bounded_cartesian_product(left, right)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("left", "right"), code=exc.type, message=exc.message()
        ) from None


def representation_profile(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> RepresentationProfileResult:
    """Compute ``r_{A+B}(x)`` for every sum ``x``."""
    try:
        _require_bounded_cartesian_product(left, right)
    except PydanticCustomError as exc:
        raise OperationDomainValidationError(
            location=("left", "right"), code=exc.type, message=exc.message()
        ) from None
    counts = _representation_function(_parse_set(left), _parse_set(right))
    entries = tuple(
        RepresentationProfileEntry(sum=value, multiplicity=counts[value])
        for value in _sorted_sums(counts)
    )
    return RepresentationProfileResult(left=left, right=right, entries=entries)


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
        RepresentationProfileEntry(sum=value, multiplicity=counts[value])
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
        RepresentationProfileEntry(sum=value, multiplicity=counts[value])
        for value in _sorted_sums(counts)
    )
    return AdditiveEnergyResult._from_kernel(
        left,
        right,
        sum(count * count for count in counts.values()),
        decomposition,
    )


def sumset_cardinality(
    left: FiniteIntegerSet, right: FiniteIntegerSet
) -> SumsetCardinalityResult:
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``)."""
    _require_bounded_cartesian_product(left, right)
    counts = _representation_function(_parse_set(left), _parse_set(right))
    support_values = _sorted_sums(counts)
    try:
        support = FiniteIntegerSet(elements=tuple(support_values))
    except ValidationError as exc:
        raise OperationDomainValidationError(
            location=("left", "right"),
            code="additive_combinatorics.sumset_support_not_composable",
            message="the produced support exceeds the canonical finite-set envelope",
        ) from exc
    return SumsetCardinalityResult._from_kernel(left, right, support)


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
        left=left,
        right=right,
        holds=not (collisions_sorted or missing),
        modulus=modulus,
        representatives=tuple(value for value in representatives_sorted),
        collisions=tuple(value for value in collisions_sorted),
        missing=tuple(missing),
    )


def verify_representation_profile(result: RepresentationProfileResult) -> bool:
    """Verify a serialized representation profile against its source sets."""
    try:
        expected = representation_profile(result.left, result.right)
        return expected.entries == result.entries
    except Exception:
        return False


def verify_additive_energy(result: AdditiveEnergyResult) -> bool:
    """Verify additive energy and its decomposition against source sets."""
    try:
        expected = additive_energy(result.left, result.right)
        return (
            expected.energy == result.energy
            and expected.decomposition == result.decomposition
        )
    except Exception:
        return False


def verify_sumset_cardinality(result: SumsetCardinalityResult) -> bool:
    """Verify sumset support and cardinality against source sets."""
    try:
        expected = sumset_cardinality(result.left, result.right)
        return (
            expected.cardinality == result.cardinality
            and expected.support == result.support
        )
    except Exception:
        return False


def verify_direct_sum_predicate(result: DirectSumPredicateResult) -> bool:
    """Verify direct-sum diagnostics and conclusion against source sets."""
    try:
        expected = direct_sum_predicate(result.modulus, result.left, result.right)
        return expected == result
    except Exception:
        return False


def ordered_difference_profile(
    vectors: IntegerVectorSet,
) -> OrderedDifferenceProfileResult:
    """Compute the complete ordered-difference profile of a finite vector set."""
    set_size = len(vectors.vectors)
    dimension = len(vectors.vectors[0].coordinates)
    ordered_pairs = set_size * (set_size - 1)
    coordinate_work = ordered_pairs * dimension
    if coordinate_work > MAX_ORDERED_DIFFERENCE_COORDINATE_WORK:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="additive_combinatorics.ordered_difference_work_exceeded",
            message=(
                "ordered-difference subtraction exceeds the "
                f"{MAX_ORDERED_DIFFERENCE_COORDINATE_WORK:,}-coordinate work budget"
            ),
        )
    output_cells = set_size * dimension + ordered_pairs * (dimension + 2)
    if output_cells > MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:
        raise OperationDomainValidationError(
            location=("vectors",),
            code="additive_combinatorics.ordered_difference_output_exceeded",
            message=(
                "ordered-difference profile exceeds the "
                f"{MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:,}-cell result bound"
            ),
        )
    vector_values = [vector.as_int_tuple() for vector in vectors.vectors]
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


def _admit_ordered_difference_entry(
    entry: OrderedDifferenceEntry,
    *,
    set_size: int,
    dimension: int,
) -> tuple[int, ...]:
    ordered_pairs = set_size * (set_size - 1)
    if (
        type(entry) is not OrderedDifferenceEntry
        or type(entry.difference) is not IntegerVector
        or type(entry.difference.coordinates) is not tuple
        or type(entry.pairs) is not tuple
    ):
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_entry_shape",
            message="ordered-difference rows must retain their typed tuple shape",
        )
    if len(entry.difference.coordinates) != dimension:
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_axes",
            message="ordered-difference rows must match the source dimension",
        )
    if any(
        not isinstance(coordinate, str)
        or len(coordinate) > _MAX_VECTOR_COORDINATE_LENGTH
        for coordinate in entry.difference.coordinates
    ):
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_coordinate_bound",
            message="ordered-difference rows exceed their admitted coordinate bound",
        )
    if (
        type(entry.multiplicity) is not int
        or not 0 < entry.multiplicity <= ordered_pairs
        or len(entry.pairs) > ordered_pairs
    ):
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_multiplicity_bound",
            message="ordered-difference multiplicities exceed their admitted bound",
        )
    pairs = tuple(
        _admit_ordered_difference_pair(
            pair, set_size=set_size, location=("claim", "entries", "pairs")
        )
        for pair in entry.pairs
    )
    if pairs != tuple(sorted(set(pairs))):
        raise OperationDomainValidationError(
            location=("claim", "entries", "pairs"),
            code="additive_combinatorics.ordered_difference_pair_order",
            message="ordered-difference pairs must be unique and ordered",
        )
    return entry.difference.as_int_tuple()


def _admit_ordered_difference_pair(
    pair: object,
    *,
    set_size: int,
    location: tuple[str, ...],
) -> tuple[int, int]:
    if type(pair) is not OrderedDifferencePair:
        raise OperationDomainValidationError(
            location=location,
            code="additive_combinatorics.ordered_difference_pair_bounds",
            message="ordered-difference pairs must be distinct source indices",
        )
    if type(pair.left_index) is not int or type(pair.right_index) is not int:
        raise OperationDomainValidationError(
            location=location,
            code="additive_combinatorics.ordered_difference_pair_bounds",
            message="ordered-difference pair indices must be native integers",
        )
    if (
        not 0 <= pair.left_index < set_size
        or not 0 <= pair.right_index < set_size
        or pair.left_index == pair.right_index
    ):
        raise OperationDomainValidationError(
            location=location,
            code="additive_combinatorics.ordered_difference_pair_bounds",
            message="ordered-difference pairs must be distinct source indices",
        )
    return pair.left_index, pair.right_index


def _admit_ordered_difference_source(
    vectors: IntegerVectorSet,
) -> tuple[int, int]:
    if type(vectors) is not IntegerVectorSet or type(vectors.vectors) is not tuple:
        raise OperationDomainValidationError(
            location=("claim", "vectors"),
            code="additive_combinatorics.ordered_difference_source_shape",
            message="ordered-difference claim source must retain its typed tuple shape",
        )
    set_size = len(vectors.vectors)
    if not 1 <= set_size <= _MAX_VECTOR_SET_SIZE:
        raise OperationDomainValidationError(
            location=("claim", "vectors"),
            code="additive_combinatorics.ordered_difference_source_bound",
            message="ordered-difference claim source exceeds its admitted size",
        )
    dimension = len(vectors.vectors[0].coordinates)
    if not 1 <= dimension <= _MAX_DIMENSION:
        raise OperationDomainValidationError(
            location=("claim", "vectors"),
            code="additive_combinatorics.ordered_difference_dimension_bound",
            message="ordered-difference claim dimension exceeds its admitted bound",
        )
    for vector in vectors.vectors:
        if type(vector) is not IntegerVector or type(vector.coordinates) is not tuple:
            raise OperationDomainValidationError(
                location=("claim", "vectors"),
                code="additive_combinatorics.ordered_difference_source_shape",
                message="ordered-difference source vectors must retain typed tuples",
            )
        if len(vector.coordinates) != dimension:
            raise OperationDomainValidationError(
                location=("claim", "vectors"),
                code="additive_combinatorics.ordered_difference_axes",
                message="ordered-difference source vectors have inconsistent dimensions",
            )
        if any(
            not isinstance(coordinate, str)
            or len(coordinate) > _MAX_VECTOR_COORDINATE_LENGTH
            or len(coordinate.lstrip("-")) > _MAX_COORDINATE_DIGITS
            for coordinate in vector.coordinates
        ):
            raise OperationDomainValidationError(
                location=("claim", "vectors"),
                code="additive_combinatorics.ordered_difference_coordinate_bound",
                message="ordered-difference source coordinates exceed their admitted bound",
            )
    return set_size, dimension


def _admit_ordered_difference_claim(
    claim: OrderedDifferenceProfileResult,
) -> tuple[int, int]:
    """Admit a decoded claim before traversing its source or profile rows."""
    if not isinstance(claim, OrderedDifferenceProfileResult):
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_claim_type",
            message="ordered-difference verifier requires its typed result value",
        )
    vectors = claim.vectors
    set_size, dimension = _admit_ordered_difference_source(vectors)
    if claim.set_size != set_size or claim.dimension != dimension:
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_axes",
            message="ordered-difference claim axes do not match its retained source",
        )
    ordered_pairs = set_size * (set_size - 1)
    coordinate_work = ordered_pairs * dimension
    if coordinate_work > MAX_ORDERED_DIFFERENCE_COORDINATE_WORK:
        raise OperationDomainValidationError(
            location=("claim", "vectors"),
            code="additive_combinatorics.ordered_difference_work_exceeded",
            message=(
                "ordered-difference verification exceeds the "
                f"{MAX_ORDERED_DIFFERENCE_COORDINATE_WORK:,}-coordinate work budget"
            ),
        )
    output_cells = set_size * dimension + ordered_pairs * (dimension + 2)
    if output_cells > MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_output_exceeded",
            message=(
                "ordered-difference verification exceeds the "
                f"{MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:,}-cell result bound"
            ),
        )
    if type(claim.entries) is not tuple:
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_entry_shape",
            message="ordered-difference entries must retain their typed tuple shape",
        )
    if len(claim.entries) > ordered_pairs:
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_entry_bound",
            message="ordered-difference claim has too many profile rows",
        )
    differences = tuple(
        _admit_ordered_difference_entry(entry, set_size=set_size, dimension=dimension)
        for entry in claim.entries
    )
    if tuple(differences) != tuple(sorted(set(differences))):
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_entry_order",
            message="ordered-difference entries must be unique and ordered",
        )
    if type(claim.dimension) is not int or type(claim.set_size) is not int:
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_axes",
            message="ordered-difference claim axes must be native integers",
        )
    if (
        type(claim.total_ordered_pairs) is not int
        or type(claim.support_size) is not int
    ):
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_summary_shape",
            message="ordered-difference summaries must be native integers",
        )
    if (
        type(claim.max_multiplicity) is not int
        or type(claim.has_repeated_difference) is not bool
    ):
        raise OperationDomainValidationError(
            location=("claim",),
            code="additive_combinatorics.ordered_difference_summary_shape",
            message="ordered-difference summaries have an invalid typed shape",
        )
    if claim.first_collision is not None:
        _admit_ordered_difference_pair(
            claim.first_collision,
            set_size=set_size,
            location=("claim", "first_collision"),
        )
    claimed_cells = set_size * dimension + sum(
        len(entry.difference.coordinates) + 2 * len(entry.pairs)
        for entry in claim.entries
    )
    if claimed_cells > MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:
        raise OperationDomainValidationError(
            location=("claim", "entries"),
            code="additive_combinatorics.ordered_difference_output_exceeded",
            message=(
                "ordered-difference claim payload exceeds the "
                f"{MAX_ORDERED_DIFFERENCE_OUTPUT_CELLS:,}-cell result bound"
            ),
        )
    return set_size, dimension


def verify_ordered_difference_profile(
    claim: OrderedDifferenceProfileResult,
) -> bool:
    """Verify a complete profile claim against its retained vector source.

    Result decoding only checks canonical axes and bounded row shape. This
    consumer spends its own admitted pair-coordinate budget to check every
    source pair, every claimed difference row, all multiplicities, and every
    aggregate summary including completeness and the collision witness.
    """
    try:
        set_size, dimension = _admit_ordered_difference_claim(claim)
        source = tuple(vector.as_int_tuple() for vector in claim.vectors.vectors)
        if len(set(source)) != set_size:
            return False
        expected: dict[tuple[int, ...], list[tuple[int, int]]] = {}
        for left_index, left in enumerate(source):
            for right_index, right in enumerate(source):
                if left_index == right_index:
                    continue
                difference = tuple(
                    left[index] - right[index] for index in range(dimension)
                )
                expected.setdefault(difference, []).append((left_index, right_index))

        expected_entries = tuple(
            (difference, len(pairs), tuple(pairs))
            for difference, pairs in sorted(expected.items())
        )
        actual_entries = tuple(
            (
                entry.difference.as_int_tuple(),
                entry.multiplicity,
                tuple((pair.left_index, pair.right_index) for pair in entry.pairs),
            )
            for entry in claim.entries
        )
        if actual_entries != expected_entries:
            return False

        total = sum(len(pairs) for pairs in expected.values())
        maximum = max((len(pairs) for pairs in expected.values()), default=0)
        first_collision = next(
            (
                pairs[0]
                for difference, pairs in sorted(expected.items())
                if len(pairs) > 1
            ),
            None,
        )
        expected_collision = (
            None
            if first_collision is None
            else OrderedDifferencePair(
                left_index=first_collision[0], right_index=first_collision[1]
            )
        )
        return (
            claim.total_ordered_pairs == total
            and claim.support_size == len(expected)
            and claim.max_multiplicity == maximum
            and claim.has_repeated_difference == (maximum > 1)
            and claim.first_collision == expected_collision
            and total == set_size * (set_size - 1)
        )
    except Exception:
        return False


__all__ = [
    "additive_energy",
    "direct_sum_predicate",
    "multiset_sum_representation_profile",
    "ordered_difference_profile",
    "representation_profile",
    "subset_sum_profile",
    "sumset_cardinality",
    "verify_ordered_difference_profile",
]
