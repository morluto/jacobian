"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

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


class FiniteIntegerVectorSet(StrictModel):
    """One finite set of distinct integer vectors in Z^d."""

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


class OrderedDifferenceProfileResult(StrictModel):
    """The complete exact ordered-difference profile of a finite integer-vector set."""

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
        expected_total = self.set_size * (self.set_size - 1)
        if self.total_ordered_pairs != expected_total:
            raise ValueError("total_ordered_pairs must equal set_size * (set_size - 1)")
        if self.support_size != len(self.classes):
            raise ValueError("support_size must match the class count")
        class_total = sum(c.multiplicity for c in self.classes)
        if class_total != expected_total:
            raise ValueError("class multiplicities must sum to total_ordered_pairs")
        if self.classes:
            computed_max = max(c.multiplicity for c in self.classes)
            if self.max_multiplicity != computed_max:
                raise ValueError("max_multiplicity must be the maximum class multiplicity")
        if self.has_repeated_difference != (self.max_multiplicity > 1):
            raise ValueError("has_repeated_difference must agree with max_multiplicity > 1")
        for cls in self.classes:
            if cls.multiplicity != len(cls.source_pairs):
                raise ValueError("multiplicity must match the source pair count")
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
