"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

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
# Ordered-difference profile for integer-vector sets
# ---------------------------------------------------------------------------

_MAX_VECTOR_DIMENSION = 8
_MAX_VECTOR_SET_SIZE = 64
_MAX_VECTOR_COORDINATE_DIGITS = 64
# A difference coordinate can grow by one digit (sum of two bounded integers).
_MAX_VECTOR_DIFFERENCE_DIGITS = _MAX_VECTOR_COORDINATE_DIGITS + 1
_MAX_ORDERED_PAIRS = _MAX_VECTOR_SET_SIZE * (_MAX_VECTOR_SET_SIZE - 1)


class IntegerVector(StrictModel):
    """One integer vector in a bounded common dimension."""

    coordinates: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=_MAX_VECTOR_DIMENSION,
    )

    @model_validator(mode="after")
    def require_bounded_coordinates(self) -> Self:
        for value in self.coordinates:
            if len(value.lstrip("-")) > _MAX_VECTOR_COORDINATE_DIGITS:
                raise ValueError("vector coordinate exceeds the digit bound")
        return self


class IntegerVectorSet(StrictModel):
    """A finite set of distinct integer vectors in a fixed dimension."""

    vectors: tuple[IntegerVector, ...] = Field(
        min_length=1,
        max_length=_MAX_VECTOR_SET_SIZE,
    )

    @model_validator(mode="after")
    def require_uniform_and_distinct(self) -> Self:
        if not self.vectors:
            return self
        dim = len(self.vectors[0].coordinates)
        for vec in self.vectors[1:]:
            if len(vec.coordinates) != dim:
                raise ValueError("all vectors must share the same dimension")
        seen: set[tuple[int, ...]] = set()
        for vec in self.vectors:
            key = tuple(parse_canonical_integer(c) for c in vec.coordinates)
            if key in seen:
                raise ValueError("vector set elements must be distinct")
            seen.add(key)
        return self


class OrderedDifferenceProfileRequest(StrictModel):
    """Compute the complete ordered-difference profile ``r_{A-A}`` of one set."""

    vectors: IntegerVectorSet


class OrderedDifferencePair(StrictModel):
    """One ordered source pair realizing a difference vector."""

    minuend_index: int = Field(ge=0)
    subtrahend_index: int = Field(ge=0)


class OrderedDifferenceClass(StrictModel):
    """One nonzero difference vector and every ordered pair realizing it."""

    difference: tuple[CanonicalInteger, ...] = Field(min_length=1)
    pairs: tuple[OrderedDifferencePair, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_nonzero_difference(self) -> Self:
        if all(parse_canonical_integer(c) == 0 for c in self.difference):
            raise ValueError("the zero difference class is not reported")
        if any(
            len(c.lstrip("-")) > _MAX_VECTOR_DIFFERENCE_DIGITS for c in self.difference
        ):
            raise ValueError("difference coordinate exceeds the digit bound")
        for pair in self.pairs:
            if pair.minuend_index == pair.subtrahend_index:
                raise ValueError("an ordered difference pair must be distinct")
        return self


class OrderedDifferenceProfileResult(StrictModel):
    """Complete ordered-difference profile of a bounded integer-vector set."""

    dimension: int = Field(ge=1, le=_MAX_VECTOR_DIMENSION)
    set_size: int = Field(ge=1, le=_MAX_VECTOR_SET_SIZE)
    classes: tuple[OrderedDifferenceClass, ...] = Field(
        max_length=_MAX_ORDERED_PAIRS,
    )
    ordered_pair_count: int = Field(ge=0)
    support_size: int = Field(ge=0)
    max_multiplicity: int = Field(ge=0)
    has_repeated_difference: bool
    first_repeated_difference: tuple[CanonicalInteger, ...] | None = Field(
        default=None,
    )

    @model_validator(mode="after")
    def require_consistent_profile(self) -> Self:
        expected_pairs = self.set_size * (self.set_size - 1)
        total_pairs = sum(len(cls.pairs) for cls in self.classes)
        if total_pairs != expected_pairs:
            raise ValueError(
                "ordered pair total must equal set_size * (set_size - 1)",
            )
        if self.ordered_pair_count != expected_pairs:
            raise ValueError("ordered_pair_count must equal the realized pair total")
        if self.support_size != len(self.classes):
            raise ValueError("support_size must equal the number of difference classes")
        max_mult = max((len(cls.pairs) for cls in self.classes), default=0)
        if self.max_multiplicity != max_mult:
            raise ValueError("max_multiplicity must equal the largest class size")
        if self.has_repeated_difference and self.first_repeated_difference is None:
            raise ValueError(
                "a repeated difference must supply a first repeated difference witness",
            )
        if (
            not self.has_repeated_difference
            and self.first_repeated_difference is not None
        ):
            raise ValueError(
                "first_repeated_difference must be absent when no difference repeats",
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
