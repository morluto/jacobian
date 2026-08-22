"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import parse_canonical_integer

_MAX_SET_SIZE = 256
_MAX_RESULT_SIZE = _MAX_SET_SIZE * _MAX_SET_SIZE


def _sorted_canonical_integers(
    values: Iterable[str],
) -> tuple[str, ...]:
    """Return canonical integers in numeric order."""
    return tuple(sorted(set(values), key=parse_canonical_integer))


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



# ---------------------------------------------------------------------------
# Ordered-difference profile for finite integer-vector sets
# ---------------------------------------------------------------------------

MAX_VECTOR_SET_SIZE = 128
MAX_VECTOR_DIMENSION = 8
# Coordinates carry at most 64 decimal digits like the canonical integers of
# the sibling set operations; an ordered difference of two admitted
# coordinates reaches 10**(MAX_VECTOR_COORDINATE_DIGITS + 1) in magnitude.
MAX_VECTOR_COORDINATE_DIGITS = 64
_MAX_COORDINATE_MAGNITUDE = 10**MAX_VECTOR_COORDINATE_DIGITS
MAX_DIFFERENCE_COORDINATE_DIGITS = MAX_VECTOR_COORDINATE_DIGITS + 1
_MAX_DIFFERENCE_MAGNITUDE = 10**MAX_DIFFERENCE_COORDINATE_DIGITS


class FiniteIntegerVectorSet(StrictModel):
    """One finite set of distinct integer vectors in Z^d.

    Every coordinate carries at most ``MAX_VECTOR_COORDINATE_DIGITS`` decimal
    digits so profile construction and result replay stay bounded.
    """

    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=MAX_VECTOR_SET_SIZE)

    @model_validator(mode="after")
    def require_valid_vectors(self) -> Self:
        if not self.vectors:
            raise ValueError("vector set must be nonempty")
        dim = len(self.vectors[0])
        if dim < 1:
            raise ValueError("vectors must have at least one coordinate")
        if dim > MAX_VECTOR_DIMENSION:
            raise ValueError(f"vector dimension exceeds the {MAX_VECTOR_DIMENSION}-coordinate bound")
        if any(len(v) != dim for v in self.vectors):
            raise ValueError("all vectors must have the same dimension")
        if any(any(type(c) is not int for c in v) for v in self.vectors):
            raise ValueError("vector coordinates must be integers")
        if any(
            abs(c) >= _MAX_COORDINATE_MAGNITUDE
            for v in self.vectors
            for c in v
        ):
            raise ValueError(
                "vector coordinates exceed the "
                f"{MAX_VECTOR_COORDINATE_DIGITS}-digit bound"
            )
        if len(set(self.vectors)) != len(self.vectors):
            raise ValueError("vectors must be distinct")
        return self


class OrderedDifferenceProfileRequest(StrictModel):
    """Compute the complete ordered-difference profile r_{A-A}(v)."""

    vectors: FiniteIntegerVectorSet


class DifferenceClassEntry(StrictModel):
    """One nonzero difference vector and its ordered source pairs."""

    difference: tuple[int, ...]
    multiplicity: int = Field(gt=0)
    source_pairs: tuple[tuple[int, int], ...]

    @model_validator(mode="after")
    def require_bounded_difference(self) -> Self:
        if any(abs(c) >= _MAX_DIFFERENCE_MAGNITUDE for c in self.difference):
            raise ValueError(
                "difference coordinates exceed the "
                f"{MAX_DIFFERENCE_COORDINATE_DIGITS}-digit bound"
            )
        return self


def _require_source_binding(result: OrderedDifferenceProfileResult) -> None:
    """Dimension and size must describe the retained source vector set."""
    if result.dimension != len(result.source_set.vectors[0]) or result.set_size != len(
        result.source_set.vectors
    ):
        raise ValueError(
            "dimension and set_size must match the retained source set"
        )


def _require_honest_repetition(
    result: OrderedDifferenceProfileResult,
    repeated: list[tuple[int, ...]],
) -> None:
    """Repetition claims must match the exact replayed difference multiset."""
    if result.has_repeated_difference != bool(repeated):
        raise ValueError("has_repeated_difference must match an exact replay")
    canonical_first = min(repeated) if repeated else None
    if result.first_repeated_difference != canonical_first:
        raise ValueError(
            "first_repeated_difference must be the lexicographically "
            "first repeated difference when repetition exists and null "
            "otherwise"
        )


def _exact_difference_counts(
    vectors: tuple[tuple[int, ...], ...],
) -> Counter[tuple[int, ...]]:
    """Exact multiset of ordered differences of a finite integer-vector set."""
    counts: Counter[tuple[int, ...]] = Counter()
    for i, vi in enumerate(vectors):
        for j, vj in enumerate(vectors):
            if i == j:
                continue
            diff = tuple(b - a for a, b in zip(vi, vj, strict=True))
            counts[diff] += 1
    return counts


def _require_class_aggregates(result: OrderedDifferenceProfileResult) -> None:
    """Aggregate fields must describe one exact partition of the ordered pairs."""
    expected_total = result.set_size * (result.set_size - 1)
    if result.total_ordered_pairs != expected_total:
        raise ValueError("total_ordered_pairs must equal set_size * (set_size - 1)")
    if result.support_size != len(result.classes):
        raise ValueError("support_size must match the class count")
    class_total = sum(c.multiplicity for c in result.classes)
    if class_total != expected_total:
        raise ValueError("class multiplicities must sum to total_ordered_pairs")
    computed_max = max((c.multiplicity for c in result.classes), default=0)
    if result.max_multiplicity != computed_max:
        raise ValueError("max_multiplicity must be the maximum class multiplicity")
    if result.has_repeated_difference != (computed_max > 1):
        raise ValueError("has_repeated_difference must agree with max_multiplicity > 1")
    for cls in result.classes:
        if cls.multiplicity != len(cls.source_pairs):
            raise ValueError("multiplicity must match the source pair count")


def _require_complete_pair_coverage(
    result: OrderedDifferenceProfileResult,
) -> None:
    """Verify every ordered source pair is present exactly once and produces
    its claimed difference via the retained source vectors."""
    seen: set[tuple[int, int]] = set()
    for cls in result.classes:
        if len(cls.difference) != result.dimension:
            raise ValueError("difference dimension must match the source set")
        for a, b in cls.source_pairs:
            if not (0 <= a < result.set_size and 0 <= b < result.set_size):
                raise ValueError("source pair index out of range")
            if a == b:
                raise ValueError("source pair must be ordered distinct indices")
            if (a, b) in seen:
                raise ValueError("source pairs must be unique across classes")
            seen.add((a, b))
            actual = tuple(
                result.source_set.vectors[a][d] - result.source_set.vectors[b][d]
                for d in range(result.dimension)
            )
            if actual != cls.difference:
                raise ValueError("source pair does not produce the claimed difference")
    if len(seen) != result.total_ordered_pairs:
        raise ValueError("source pair coverage must equal total_ordered_pairs")
    expected_pairs = {
        (i, j)
        for i in range(result.set_size)
        for j in range(result.set_size)
        if i != j
    }
    if seen != expected_pairs:
        raise ValueError("source pair set must be the complete ordered pair set")


class OrderedDifferenceProfileResult(StrictModel):
    """The complete exact ordered-difference profile of a finite integer-vector set.

    Retains the canonical source vector set so every difference class replays
    against the vectors it claims to describe.
    """

    source_set: FiniteIntegerVectorSet
    dimension: int = Field(ge=1)
    set_size: int = Field(ge=1)
    total_ordered_pairs: int = Field(ge=0)
    support_size: int = Field(ge=0)
    max_multiplicity: int = Field(ge=0)
    has_repeated_difference: bool
    first_repeated_difference: tuple[int, ...] | None = None
    classes: tuple[DifferenceClassEntry, ...]

    @model_validator(mode="after")
    def require_profile_invariants(self) -> Self:
        _require_source_binding(self)
        truth = _exact_difference_counts(self.source_set.vectors)
        claimed = {cls.difference: cls.multiplicity for cls in self.classes}
        if claimed != dict(truth):
            raise ValueError(
                "difference classes must replay against the source vectors"
            )
        repeated = [d for d, count in truth.items() if count > 1]
        _require_honest_repetition(self, repeated)
        _require_class_aggregates(self)
        _require_complete_pair_coverage(self)
        return self


__all__ = [
    "AdditiveEnergyRequest",
    "AdditiveEnergyResult",
    "DifferenceClassEntry",
    "DirectSumPredicateRequest",
    "DirectSumPredicateResult",
    "FiniteCyclicGroup",
    "FiniteIntegerSet",
    "FiniteIntegerVectorSet",
    "OrderedDifferenceProfileRequest",
    "OrderedDifferenceProfileResult",
    "RepresentationProfileEntry",
    "RepresentationProfileRequest",
    "RepresentationProfileResult",
    "SumsetCardinalityRequest",
    "SumsetCardinalityResult",
]
