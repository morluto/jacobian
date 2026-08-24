"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.additive_combinatorics.operations import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
    _subset_sum_profile_envelope,
)
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_PROFILE_ENTRIES,
    IndexedIntegerSequence,
)

_MAX_SET_SIZE = 4096
_MAX_RESULT_SIZE = _MAX_SET_SIZE * _MAX_SET_SIZE
_MAX_DIMENSION = 8
_MAX_COORDINATE_DIGITS = 6

# One serialized ordered-difference entry carries an eight-coordinate signed
# difference, a multiplicity, and one index pair, which stays under 256
# canonical JSON bytes even at the widest admitted coordinates, and a complete
# profile holds n*(n-1) entries. Admitting only set sizes whose worst-case
# entry array fits the 4 MiB result budget keeps the full exact result safely
# inside Jacobian's 10 MiB canonical output limit once the retained sources,
# scalar header, and operation envelope are included.
_MAX_ENTRY_WIRE_BYTES = 256
_MAX_PROFILE_RESULT_BUDGET_BYTES = 4 * 1024 * 1024
_MAX_VECTOR_SET_SIZE = (
    math.isqrt(4 * (_MAX_PROFILE_RESULT_BUDGET_BYTES // _MAX_ENTRY_WIRE_BYTES) + 1) + 1
) // 2
_MAX_TOTAL_ORDERED_PAIRS = _MAX_VECTOR_SET_SIZE * (_MAX_VECTOR_SET_SIZE - 1)

# A request coordinate carries at most six digits plus an optional sign; an
# exact difference of two such coordinates grows by at most one digit, so any
# vector coordinate string carries at most seven digits plus an optional sign.
# The schema-level character ceiling keeps every later integer conversion
# bounded before any bigint is constructed.
_MAX_VECTOR_COORDINATE_LENGTH = _MAX_COORDINATE_DIGITS + 2

CanonicalVectorCoordinate = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=_MAX_VECTOR_COORDINATE_LENGTH,
        strict=True,
    ),
]


def _sorted_canonical_integers(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Return canonical integers in numeric order."""
    return tuple(sorted(set(values), key=parse_canonical_integer))


def _require_bounded_coordinate(value: str, label: str) -> None:
    digits = len(value.lstrip("-"))
    if digits > _MAX_COORDINATE_DIGITS:
        raise ValueError(
            f"{label} exceeds the {_MAX_COORDINATE_DIGITS}-digit coordinate bound"
        )


class IntegerVector(StrictModel):
    """One integer vector in a bounded common dimension.

    This is the domain's canonical vector value: request sources, retained
    sources, and difference entries all carry it so exact vectors compose
    downstream without reconstruction. Each coordinate carries at most seven
    digits plus an optional sign, enforced at the string level so oversized
    values are rejected before any integer conversion.
    """

    coordinates: tuple[CanonicalVectorCoordinate, ...] = Field(
        min_length=1,
        max_length=_MAX_DIMENSION,
        description=(
            "Canonical integer coordinates sharing one dimension in "
            f"[1, {_MAX_DIMENSION}], each carrying at most "
            f"{_MAX_VECTOR_COORDINATE_LENGTH - 1} digits plus an optional sign."
        ),
        examples=[("0", "1")],
    )

    def as_int_tuple(self) -> tuple[int, ...]:
        return tuple(parse_canonical_integer(c) for c in self.coordinates)


def _vector_from_ints(values: tuple[int, ...]) -> IntegerVector:
    return IntegerVector(coordinates=tuple(format_canonical_integer(v) for v in values))


class IntegerVectorSet(StrictModel):
    """A finite set of distinct integer vectors in a fixed dimension.

    Coordinates carry at most six decimal digits; the listed order is the
    index order used by ordered-difference pairs.
    """

    vectors: tuple[IntegerVector, ...] = Field(
        min_length=1,
        max_length=_MAX_VECTOR_SET_SIZE,
    )

    @model_validator(mode="after")
    def require_uniform_distinct_bounded(self) -> Self:
        dimension = len(self.vectors[0].coordinates)
        seen: set[tuple[int, ...]] = set()
        for vec in self.vectors:
            for coordinate in vec.coordinates:
                _require_bounded_coordinate(coordinate, "vector coordinate")
            values = vec.as_int_tuple()
            if len(values) != dimension:
                raise ValueError("all vectors must share the same dimension")
            if not 1 <= dimension <= _MAX_DIMENSION:
                raise ValueError(
                    f"vector dimension must be between 1 and {_MAX_DIMENSION}"
                )
            if values in seen:
                raise ValueError("vectors must be unique")
            seen.add(values)
        return self


def _check_totals(
    entries: tuple[OrderedDifferenceEntry, ...],
    total_ordered_pairs: int,
    set_size: int,
    support_size: int,
) -> None:
    total = sum(entry.multiplicity for entry in entries)
    if total != total_ordered_pairs:
        raise ValueError("total ordered pairs must match sum of multiplicities")
    if total_ordered_pairs != set_size * (set_size - 1):
        raise ValueError("total_ordered_pairs must equal set_size*(set_size-1)")
    if support_size != len(entries):
        raise ValueError("support_size must equal the number of entries")


def _check_max_and_repeated(
    entries: tuple[OrderedDifferenceEntry, ...],
    max_multiplicity: int,
    has_repeated_difference: bool,
    first_collision: OrderedDifferencePair | None,
) -> None:
    if not entries:
        if max_multiplicity != 0:
            raise ValueError("max_multiplicity must be 0 when entries is empty")
    elif max_multiplicity != max(e.multiplicity for e in entries):
        raise ValueError("max_multiplicity must be the maximum entry multiplicity")
    expected_repeated = (max_multiplicity > 1) if entries else False
    if has_repeated_difference != expected_repeated:
        raise ValueError("has_repeated_difference must match max_multiplicity > 1")
    if has_repeated_difference and first_collision is None:
        raise ValueError("first_collision must be present when has_repeated_difference")
    if not has_repeated_difference and first_collision is not None:
        raise ValueError(
            "first_collision must be null when has_repeated_difference is false"
        )


def _check_entries_sorted(entries: tuple[OrderedDifferenceEntry, ...]) -> None:
    diffs = [entry.difference.as_int_tuple() for entry in entries]
    if diffs != sorted(diffs):
        raise ValueError("entries must be sorted by difference")
    if len(set(diffs)) != len(diffs):
        raise ValueError("entries differences must be unique")


def _check_entry_pairs(
    entries: tuple[OrderedDifferenceEntry, ...],
    dimension: int,
    vectors: tuple[IntegerVector, ...],
    set_size: int,
) -> None:
    zero = tuple(0 for _ in range(dimension))
    for entry in entries:
        difference = entry.difference.as_int_tuple()
        if len(difference) != dimension:
            raise ValueError("entry difference dimension must match result dimension")
        if difference == zero:
            raise ValueError("entry difference must be nonzero")
        previous: tuple[int, int] | None = None
        for pair in entry.pairs:
            key = (pair.left_index, pair.right_index)
            if previous is not None and key <= previous:
                raise ValueError(
                    "entry pairs must be sorted and unique in lexicographic order"
                )
            previous = key
            if pair.left_index >= set_size or pair.right_index >= set_size:
                raise ValueError("pair indices must be less than set_size")
            if vectors:
                expected = tuple(
                    vectors[pair.left_index].as_int_tuple()[k]
                    - vectors[pair.right_index].as_int_tuple()[k]
                    for k in range(dimension)
                )
                if expected != difference:
                    raise ValueError("pair difference must match vectors")


def _check_all_pairs_exactly_once(
    entries: tuple[OrderedDifferenceEntry, ...],
    set_size: int,
) -> None:
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        for pair in entry.pairs:
            key = (pair.left_index, pair.right_index)
            if key in seen:
                raise ValueError(f"ordered pair {key} appears more than once")
            seen.add(key)
    expected: set[tuple[int, int]] = {
        (i, j) for i in range(set_size) for j in range(set_size) if i != j
    }
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(
            f"entries must contain every ordered pair exactly once; "
            f"missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}"
        )


def _check_first_collision(
    entries: tuple[OrderedDifferenceEntry, ...],
    has_repeated_difference: bool,
    first_collision: OrderedDifferencePair | None,
) -> None:
    if entries and has_repeated_difference:
        # Pair order is canonically lexicographic (checked in
        # _check_entry_pairs), so pairs[0] of an entry is independently
        # determined as its minimum pair; the witness must be exactly that
        # designated pair of the first sorted repeated-difference entry.
        expected_entry = next((e for e in entries if e.multiplicity > 1), None)
        if expected_entry is None:
            raise ValueError(
                "has_repeated_difference requires a repeated difference entry"
            )
        if first_collision != expected_entry.pairs[0]:
            raise ValueError(
                "first_collision must be the designated pair of the first "
                "repeated-difference entry"
            )
    elif not entries and first_collision is not None:
        raise ValueError("first_collision must be null when entries is empty")


class FiniteIntegerSet(StrictModel):
    """One finite set of canonical integers, possibly empty."""

    elements: tuple[CanonicalInteger, ...] = Field(max_length=_MAX_SET_SIZE)

    @model_validator(mode="after")
    def require_unique_elements(self) -> Self:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("finite set elements must be unique")
        return self


class FiniteCyclicGroup(StrictModel):
    """The cyclic group ``Z_n`` carrying a direct-sum/tiling predicate."""

    modulus: int = Field(gt=1, le=_MAX_RESULT_SIZE)

    @model_validator(mode="after")
    def require_valid_modulus(self) -> Self:
        if self.modulus < 2:
            raise ValueError("cyclic group modulus must be at least 2")
        return self


# ---------------------------------------------------------------------------
# Representation profile
# ---------------------------------------------------------------------------


class RepresentationProfileRequest(StrictModel):
    """Compute ``r_{A+B}(x)`` for every sum ``x`` of two finite integer sets."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class RepresentationProfileEntry(StrictModel):
    """One sum and its representation multiplicity."""

    sum: CanonicalInteger
    multiplicity: int = Field(gt=0)


class RepresentationProfileResult(StrictModel):
    """Support and multiplicities of the representation function ``r_{A+B}``."""

    entries: tuple[RepresentationProfileEntry, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_entries(self) -> Self:
        sums = tuple(entry.sum for entry in self.entries)
        if tuple(sums) != _sorted_canonical_integers(sums):
            raise ValueError(
                "representation profile sums must be sorted and unique",
            )
        if any(entry.multiplicity <= 0 for entry in self.entries):
            raise ValueError("representation multiplicities must be positive")
        return self


# ---------------------------------------------------------------------------
# Complete indexed subset-sum profile
# ---------------------------------------------------------------------------


class SubsetSumProfileRequest(StrictModel):
    """Compute every indexed subset-sum multiplicity, including the empty set.

    Admission is result-sensitive: it bounds the exact sum span, the number of
    source-selection vectors, sparse-DP transitions, multiplicity digits, and
    worst-case serialized profile before the dynamic program begins.
    """

    source: IndexedIntegerSequence = Field(
        description=(
            "The ordered indexed integer sequence. Each position is selectable "
            "at most once; repeated values and zeros remain distinct positions. "
            "Before execution, S=min(2^n, positive_sum-negative_sum+1, "
            "product(m_v+1) over distinct nonzero values v) must fit "
            f"{MAX_SUBSET_SUM_PROFILE_ENTRIES:,} "
            f"rows, 4*n*S must not exceed {MAX_SUBSET_SUM_DP_TRANSITIONS:,} "
            "dictionary transitions across construction and validation replay, "
            "and the conservative serialized-result estimate must not exceed "
            f"{MAX_SUBSET_SUM_PROFILE_RESULT_BYTES:,} bytes."
        ),
    )

    @model_validator(mode="after")
    def require_admitted_profile_envelope(self) -> Self:
        _subset_sum_profile_envelope(self.source)
        return self


# ---------------------------------------------------------------------------
# Additive energy
# ---------------------------------------------------------------------------


class AdditiveEnergyRequest(StrictModel):
    """Compute the additive energy ``E(A, B) = sum_x r_{A+B}(x)^2``."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class AdditiveEnergyResult(StrictModel):
    """Exact additive energy and its decomposition by sum."""

    energy: int = Field(ge=0)
    decomposition: tuple[RepresentationProfileEntry, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_decomposition(self) -> Self:
        sums = tuple(entry.sum for entry in self.decomposition)
        if tuple(sums) != _sorted_canonical_integers(sums):
            raise ValueError("additive energy sums must be sorted and unique")
        if any(entry.multiplicity <= 0 for entry in self.decomposition):
            raise ValueError("additive energy multiplicities must be positive")
        if self.energy != sum(entry.multiplicity**2 for entry in self.decomposition):
            raise ValueError(
                "additive energy must equal the sum of squared multiplicities",
            )
        return self


# ---------------------------------------------------------------------------
# Sumset cardinality
# ---------------------------------------------------------------------------


class SumsetCardinalityRequest(StrictModel):
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``)."""

    left: FiniteIntegerSet
    right: FiniteIntegerSet


class SumsetCardinalityResult(StrictModel):
    """Cardinality of the sumset and its sorted support."""

    cardinality: int = Field(ge=0)
    support: tuple[CanonicalInteger, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_support(self) -> Self:
        sums = list(self.support)
        if tuple(sums) != _sorted_canonical_integers(sums):
            raise ValueError("sumset support must be sorted and unique")
        if self.cardinality != len(self.support):
            raise ValueError("cardinality must equal the support length")
        return self


# ---------------------------------------------------------------------------
# Direct sum / tiling predicate in Z_n
# ---------------------------------------------------------------------------


class DirectSumPredicateRequest(StrictModel):
    """Decide whether ``A (\\oplus) B = Z_n`` inside a finite cyclic group."""

    modulus: int = Field(gt=1, le=_MAX_RESULT_SIZE)
    left: FiniteIntegerSet
    right: FiniteIntegerSet


class DirectSumPredicateResult(StrictModel):
    """Whether the direct sum tiles ``Z_n`` and witnesses/counterexamples."""

    holds: bool
    modulus: int = Field(gt=1)
    representatives: tuple[CanonicalInteger, ...] = Field(default=())
    collisions: tuple[CanonicalInteger, ...] = Field(default=())
    missing: tuple[CanonicalInteger, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical_diagnostics(self) -> Self:
        for name in ("collisions", "missing"):
            values = [parse_canonical_integer(value) for value in getattr(self, name)]
            if values != sorted(set(values)):
                raise ValueError(
                    f"direct-sum {name} values must be sorted and unique",
                )
        return self


__all__ = [
    "AdditiveEnergyRequest",
    "AdditiveEnergyResult",
    "DirectSumPredicateRequest",
    "DirectSumPredicateResult",
    "FiniteCyclicGroup",
    "FiniteIntegerSet",
    "OrderedDifferenceEntry",
    "OrderedDifferencePair",
    "OrderedDifferenceProfileRequest",
    "OrderedDifferenceProfileResult",
    "RepresentationProfileEntry",
    "RepresentationProfileRequest",
    "RepresentationProfileResult",
    "SubsetSumProfileRequest",
    "SumsetCardinalityRequest",
    "SumsetCardinalityResult",
]


class OrderedDifferenceProfileRequest(StrictModel):
    """Compute the ordered-difference profile r_{A-A}(v) for a finite set in Z^d.

    Vectors must be distinct, share a common dimension 1..8, and each coordinate
    is bounded to 6 digits in magnitude. The set size is derived from the
    worst-case serialized result so the complete profile always fits within
    Jacobian's canonical output budget.
    """

    vectors: IntegerVectorSet = Field(
        description=(
            "Finite set of distinct integer vectors in Z^d with 1<=d<=8, each "
            "coordinate bounded to at most 6 digits in magnitude (abs value "
            f"<10^{_MAX_COORDINATE_DIGITS}), all vectors share the same dimension, "
            f"and vector entries are unique; set size at most {_MAX_VECTOR_SET_SIZE}."
        ),
    )


class OrderedDifferencePair(StrictModel):
    """One ordered source pair (i, j) with i != j."""

    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)


class OrderedDifferenceEntry(StrictModel):
    """One nonzero difference vector and its ordered source pairs.

    ``difference`` is the domain's canonical ``IntegerVector`` value so exact
    differences compose downstream without reconstruction; difference
    coordinates may carry one more digit than request coordinates because a
    difference of two bounded integers can grow by one digit.
    """

    difference: IntegerVector
    multiplicity: int = Field(gt=0)
    pairs: tuple[OrderedDifferencePair, ...] = Field(default=())

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        if self.multiplicity != len(self.pairs):
            raise ValueError("multiplicity must equal the number of pairs")
        for pair in self.pairs:
            if pair.left_index == pair.right_index:
                raise ValueError("pair indices must be distinct")
        return self


class OrderedDifferenceProfileResult(StrictModel):
    """Complete ordered-difference profile for a finite set in Z^d."""

    vectors: IntegerVectorSet
    dimension: int = Field(ge=1, le=_MAX_DIMENSION)
    set_size: int = Field(ge=1, le=_MAX_VECTOR_SET_SIZE)
    total_ordered_pairs: int = Field(ge=0, le=_MAX_TOTAL_ORDERED_PAIRS)
    support_size: int = Field(ge=0, le=_MAX_TOTAL_ORDERED_PAIRS)
    max_multiplicity: int = Field(ge=0, le=_MAX_TOTAL_ORDERED_PAIRS)
    entries: tuple[OrderedDifferenceEntry, ...] = Field(default=())
    has_repeated_difference: bool = False
    first_collision: OrderedDifferencePair | None = None

    @model_validator(mode="after")
    def require_vectors(self) -> Self:
        # The canonical vector-set value enforces distinctness, uniform
        # bounded dimension, and nonemptiness; replay binds the remaining
        # derived fields to it.
        if len(self.vectors.vectors) != self.set_size:
            raise ValueError("vectors length must equal set_size")
        if len(self.vectors.vectors[0].coordinates) != self.dimension:
            raise ValueError("dimension must match vectors")
        return self

    @model_validator(mode="after")
    def require_totals(self) -> Self:
        _check_totals(
            self.entries, self.total_ordered_pairs, self.set_size, self.support_size
        )
        return self

    @model_validator(mode="after")
    def require_max_and_repeated(self) -> Self:
        _check_max_and_repeated(
            self.entries,
            self.max_multiplicity,
            self.has_repeated_difference,
            self.first_collision,
        )
        return self

    @model_validator(mode="after")
    def require_entries(self) -> Self:
        if self.entries:
            _check_entries_sorted(self.entries)
            _check_entry_pairs(
                self.entries, self.dimension, self.vectors.vectors, self.set_size
            )
            _check_all_pairs_exactly_once(self.entries, self.set_size)
            _check_first_collision(
                self.entries, self.has_repeated_difference, self.first_collision
            )
        elif self.first_collision is not None:
            raise ValueError("first_collision must be null when entries is empty")
        return self
