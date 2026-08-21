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
    "SumsetCardinalityRequest",
    "SumsetCardinalityResult",
]


class OrderedDifferenceProfileRequest(StrictModel):
    """Compute the ordered-difference profile r_{A-A}(v) for a finite set in Z^d."""

    vectors: tuple[tuple[int, ...], ...] = Field(min_length=1, max_length=_MAX_SET_SIZE)

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if not self.vectors:
            raise ValueError("vectors must be non-empty")
        dimension = len(self.vectors[0])
        if dimension == 0:
            raise ValueError("vectors must be non-empty")
        for vec in self.vectors:
            if len(vec) != dimension:
                raise ValueError("all vectors must have the same dimension")
        return self


class OrderedDifferencePair(StrictModel):
    """One ordered source pair (i, j) with i != j."""

    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)


class OrderedDifferenceEntry(StrictModel):
    """One nonzero difference vector and its ordered source pairs."""

    difference: tuple[int, ...]
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

    dimension: int = Field(ge=1)
    set_size: int = Field(ge=0)
    total_ordered_pairs: int = Field(ge=0)
    support_size: int = Field(ge=0)
    max_multiplicity: int = Field(ge=0)
    entries: tuple[OrderedDifferenceEntry, ...] = Field(default=())
    has_repeated_difference: bool = False
    first_collision: OrderedDifferencePair | None = None

    @model_validator(mode="after")
    def require_canonical(self) -> Self:
        total = sum(entry.multiplicity for entry in self.entries)
        if total != self.total_ordered_pairs:
            raise ValueError("total ordered pairs must match sum of multiplicities")
        if self.entries and self.max_multiplicity != max(
            e.multiplicity for e in self.entries
        ):
            raise ValueError("max_multiplicity must be the maximum entry multiplicity")
        if self.has_repeated_difference != (
            self.max_multiplicity > 1 if self.entries else False
        ):
            raise ValueError("has_repeated_difference must match max_multiplicity > 1")
        if self.has_repeated_difference and self.first_collision is None:
            raise ValueError(
                "first_collision must be present when has_repeated_difference"
            )
        return self
