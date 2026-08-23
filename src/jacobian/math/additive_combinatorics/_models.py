"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer

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
# Bounded integer work: each admitted coordinate carries at most 64 decimal
# digits and each difference of two admitted coordinates is proven to carry at
# most one more, so every profile arithmetic step and its canonical difference
# encoding stay inside a fixed, published budget. Differences are carried as
# canonical integer strings, so no derived value depends on raw JSON-number
# interoperability ranges.
BoundedCoordinate = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]{0,63})$",
        max_length=65,
        strict=True,
    ),
]
ResultCoordinate = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]{0,64})$",
        max_length=66,
        strict=True,
    ),
]

# Source coordinates are admitted one digit narrower than the canonical
# vector type: differences of two admitted sources then always remain
# representable as canonical result coordinates.
MAX_SOURCE_COORDINATE_DIGITS = 64


class IntegerVector(StrictModel):
    """One integer vector of bounded canonical decimal coordinates.

    The canonical vector type admits both admitted source coordinates (at
    most 64 decimal digits) and derived difference coordinates (at most 65
    decimal digits), so a returned difference can be carried unchanged by
    any consumer accepting the canonical type; operations that consume
    vectors as sources enforce their narrower admission in the request.
    """

    coordinates: tuple[ResultCoordinate, ...] = Field(
        min_length=1,
        max_length=MAX_VECTOR_DIMENSION,
        description="Canonical integer coordinates, at most 65 decimal digits each.",
    )

    def as_ints(self) -> tuple[int, ...]:
        return tuple(parse_canonical_integer(value) for value in self.coordinates)


class IntegerVectorSet(StrictModel):
    """One finite set of distinct integer vectors in Z^d."""

    vectors: tuple[IntegerVector, ...] = Field(
        min_length=1, max_length=MAX_VECTOR_SET_SIZE
    )

    @model_validator(mode="after")
    def require_valid_vectors(self) -> Self:
        dim = len(self.vectors[0].coordinates)
        if any(len(v.coordinates) != dim for v in self.vectors):
            raise ValueError("all vectors must have the same dimension")
        if len(set(self.vectors)) != len(self.vectors):
            raise ValueError("vectors must be distinct")
        return self


class OrderedDifferenceProfileRequest(StrictModel):
    """Compute the complete ordered-difference profile r_{A-A}(v).

    Source vectors are admitted only with the narrower 64-digit coordinates:
    differences of admitted sources then always fit the canonical 65-digit
    vector type, so every accepted request returns its typed result.
    """

    vectors: IntegerVectorSet

    @model_validator(mode="after")
    def require_source_coordinates(self) -> Self:
        for vector in self.vectors.vectors:
            for value in vector.coordinates:
                if len(value.lstrip("-")) > MAX_SOURCE_COORDINATE_DIGITS:
                    raise ValueError(
                        "source vector coordinates must carry at most "
                        f"{MAX_SOURCE_COORDINATE_DIGITS} decimal digits"
                    )
        return self


class OrderedDifferencePair(StrictModel):
    """One ordered (minuend, subtrahend) index pair."""

    minuend_index: int = Field(ge=0)
    subtrahend_index: int = Field(ge=0)


class OrderedDifferenceClass(StrictModel):
    """One nonzero difference vector and its ordered source pairs."""

    difference: IntegerVector
    pairs: tuple[OrderedDifferencePair, ...] = Field(min_length=1)


def _require_pair_replay(
    result: OrderedDifferenceProfileResult,
    points: tuple[tuple[int, ...], ...],
    dimension: int,
) -> None:
    """Aggregate counts plus per-pair replay against the retained source."""
    expected_total = result.set_size * (result.set_size - 1)
    if result.ordered_pair_count != expected_total:
        raise ValueError("ordered_pair_count must equal set_size * (set_size - 1)")
    if result.support_size != len(result.classes):
        raise ValueError("support_size must match the class count")
    # One canonically ordered class per distinct replayed difference: a
    # difference split across several classes would inflate both the class
    # count and the support size while aggregate replay still passed.
    differences = [cls.difference.as_ints() for cls in result.classes]
    if len(set(differences)) != len(differences):
        raise ValueError("each distinct difference must appear in exactly one class")
    class_total = sum(len(cls.pairs) for cls in result.classes)
    if class_total != expected_total:
        raise ValueError("pair counts must sum to ordered_pair_count")
    computed_max = max((len(cls.pairs) for cls in result.classes), default=0)
    if result.max_multiplicity != computed_max:
        raise ValueError("max_multiplicity must be the maximum class multiplicity")
    if result.has_repeated_difference != (result.max_multiplicity > 1):
        raise ValueError("has_repeated_difference must agree with max_multiplicity > 1")
    seen: set[tuple[int, int]] = set()
    for cls in result.classes:
        difference = cls.difference.as_ints()
        if len(difference) != dimension:
            raise ValueError("difference dimension must match the source set")
        _require_pair_in_class(cls, points, dimension, difference, seen)


def _require_pair_in_class(
    cls: OrderedDifferenceClass,
    points: tuple[tuple[int, ...], ...],
    dimension: int,
    difference: tuple[int, ...],
    seen: set[tuple[int, int]],
) -> None:
    """Each ordered pair replays its claimed difference exactly once."""
    for pair in cls.pairs:
        a, b = pair.minuend_index, pair.subtrahend_index
        if not (0 <= a < len(points) and 0 <= b < len(points)):
            raise ValueError("source pair index out of range")
        if a == b:
            raise ValueError("source pair must be ordered distinct indices")
        if (a, b) in seen:
            raise ValueError("source pairs must be unique across classes")
        seen.add((a, b))
        actual = tuple(points[a][d] - points[b][d] for d in range(dimension))
        if actual != difference:
            raise ValueError(
                "class difference does not equal "
                "source[minuend] minus source[subtrahend] at "
                f"ordered pair ({a}, {b})"
            )


def _require_class_coverage(
    result: OrderedDifferenceProfileResult,
    points: tuple[tuple[int, ...], ...],
) -> dict[tuple[int, ...], int]:
    """Every exact difference class must appear with its full multiplicity."""
    truth = _exact_difference_counts(points)
    claimed: dict[tuple[int, ...], int] = {}
    for cls in result.classes:
        key = cls.difference.as_ints()
        claimed[key] = claimed.get(key, 0) + len(cls.pairs)
    if claimed != dict(truth):
        raise ValueError("difference classes must replay against the source vectors")
    return truth


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


class OrderedDifferenceProfileResult(StrictModel):
    """The complete exact ordered-difference profile of a finite integer-vector set.

    Retains the canonical source vector set and encodes every difference as
    canonical integers, so each class replays against the vectors it claims
    to describe without depending on raw JSON-number range.
    """

    vectors: IntegerVectorSet
    dimension: int = Field(ge=1)
    set_size: int = Field(ge=1)
    ordered_pair_count: int = Field(ge=0)
    support_size: int = Field(ge=0)
    max_multiplicity: int = Field(ge=0)
    has_repeated_difference: bool
    first_repeated_difference: tuple[CanonicalInteger, ...] | None = None
    classes: tuple[OrderedDifferenceClass, ...]

    @model_validator(mode="after")
    def require_profile_invariants(self) -> Self:
        points = tuple(vector.as_ints() for vector in self.vectors.vectors)
        dimension = len(points[0])
        if self.dimension != dimension or self.set_size != len(points):
            raise ValueError(
                "dimension and set_size must match the retained source set"
            )
        _require_pair_replay(self, points, dimension)
        truth = _require_class_coverage(self, points)
        repeated = [diff for diff, count in truth.items() if count > 1]
        if self.has_repeated_difference != bool(repeated):
            raise ValueError("has_repeated_difference must match an exact replay")
        if self.first_repeated_difference is not None:
            numeric_first = min(repeated)
            canonical_first = tuple(format_canonical_integer(c) for c in numeric_first)
            if tuple(self.first_repeated_difference) != canonical_first:
                raise ValueError(
                    "first_repeated_difference must identify the first class "
                    "of multiplicity > 1 in exact numeric order"
                )
        else:
            if repeated:
                raise ValueError(
                    "a repeated difference must be reported when one exists"
                )
        return self


__all__ = [
    "AdditiveEnergyRequest",
    "AdditiveEnergyResult",
    "DirectSumPredicateRequest",
    "DirectSumPredicateResult",
    "FiniteCyclicGroup",
    "FiniteIntegerSet",
    "IntegerVector",
    "IntegerVectorSet",
    "OrderedDifferenceClass",
    "OrderedDifferencePair",
    "OrderedDifferenceProfileRequest",
    "OrderedDifferenceProfileResult",
    "RepresentationProfileEntry",
    "RepresentationProfileRequest",
    "RepresentationProfileResult",
    "SumsetCardinalityRequest",
    "SumsetCardinalityResult",
]
