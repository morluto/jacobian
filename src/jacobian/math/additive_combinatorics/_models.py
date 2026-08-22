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
# Ordered-difference profile for integer-vector sets
# ---------------------------------------------------------------------------

_MAX_VECTOR_DIMENSION = 8
_MAX_VECTOR_SET_SIZE = 64
_MAX_VECTOR_COORDINATE_DIGITS = 64
# A difference coordinate can grow by one digit (sum of two bounded integers).
_MAX_VECTOR_DIFFERENCE_DIGITS = _MAX_VECTOR_COORDINATE_DIGITS + 1
_MAX_ORDERED_PAIRS = _MAX_VECTOR_SET_SIZE * (_MAX_VECTOR_SET_SIZE - 1)
_VECTOR_COORDINATE_PATTERN = (
    rf"^(?:0|-?[1-9][0-9]{{0,{_MAX_VECTOR_COORDINATE_DIGITS - 1}}})$"
)

VectorCoordinate = Annotated[
    str,
    StringConstraints(
        pattern=_VECTOR_COORDINATE_PATTERN,
        max_length=_MAX_VECTOR_COORDINATE_DIGITS + 1,
        strict=True,
    ),
]


class IntegerVector(StrictModel):
    """One integer vector in a bounded common dimension.

    Each coordinate is a canonical integer with at most 64 decimal digits.
    """

    coordinates: tuple[VectorCoordinate, ...] = Field(
        min_length=1,
        max_length=_MAX_VECTOR_DIMENSION,
        description=(
            "Canonical integer coordinates sharing one dimension in "
            f"[1, {_MAX_VECTOR_DIMENSION}]; each coordinate has at most "
            f"{_MAX_VECTOR_COORDINATE_DIGITS} decimal digits."
        ),
        examples=[("0", "1")],
    )

    @model_validator(mode="after")
    def require_bounded_coordinates(self) -> Self:
        for value in self.coordinates:
            if len(value.lstrip("-")) > _MAX_VECTOR_COORDINATE_DIGITS:
                raise ValueError(
                    "vector coordinate exceeds the "
                    f"{_MAX_VECTOR_COORDINATE_DIGITS}-digit bound"
                )
        return self


class IntegerVectorSet(StrictModel):
    """A finite set of distinct integer vectors in a fixed dimension.

    Coordinates remain canonical integers with at most 64 decimal digits; the
    listed order is the index order used by ordered-difference pairs.
    """

    vectors: tuple[IntegerVector, ...] = Field(
        min_length=1,
        max_length=_MAX_VECTOR_SET_SIZE,
        description=(
            "Distinct source vectors in one shared dimension; each coordinate "
            f"has at most {_MAX_VECTOR_COORDINATE_DIGITS} decimal digits."
        ),
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
    """Compute the complete ordered-difference profile ``r_{A-A}`` of one set.

    Source vectors share one dimension in ``[1, 8]``, the set has at most 64
    distinct vectors, and each coordinate has at most 64 decimal digits.
    """

    vectors: IntegerVectorSet = Field(
        description=(
            "Distinct integer vectors in one shared dimension; each coordinate "
            f"has at most {_MAX_VECTOR_COORDINATE_DIGITS} decimal digits."
        ),
    )


class OrderedDifferencePair(StrictModel):
    """One ordered source pair realizing a difference vector."""

    minuend_index: int = Field(ge=0, le=_MAX_VECTOR_SET_SIZE - 1)
    subtrahend_index: int = Field(ge=0, le=_MAX_VECTOR_SET_SIZE - 1)


class OrderedDifferenceClass(StrictModel):
    """One nonzero difference vector and every ordered pair realizing it.

    ``difference`` is the domain's canonical ``IntegerVector`` value so exact
    differences compose downstream without reconstruction.  Difference
    coordinates may carry one more digit than request coordinates (a
    difference of two bounded integers), so this canonical use admits the
    documented 65-digit boundary rather than the 64-digit request bound.
    """

    difference: IntegerVector = Field(
        description=(
            "The nonzero difference vector as the canonical IntegerVector "
            "value; coordinates may carry one more digit than request "
            "coordinates because a difference of two bounded integers can "
            "grow by one digit."
        ),
    )
    pairs: tuple[OrderedDifferencePair, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_nonzero_difference(self) -> Self:
        if all(parse_canonical_integer(c) == 0 for c in self.difference.coordinates):
            raise ValueError("the zero difference class is not reported")
        if any(
            len(c.lstrip("-")) > _MAX_VECTOR_DIFFERENCE_DIGITS
            for c in self.difference.coordinates
        ):
            raise ValueError("difference coordinate exceeds the digit bound")
        for pair in self.pairs:
            if pair.minuend_index == pair.subtrahend_index:
                raise ValueError("an ordered difference pair must be distinct")
        return self


def _source_points(
    vectors: tuple[IntegerVector, ...],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(parse_canonical_integer(c) for c in vec.coordinates) for vec in vectors
    )


def _require_source_shape(
    source: tuple[IntegerVector, ...],
    *,
    dimension: int,
    set_size: int,
) -> tuple[tuple[int, ...], ...]:
    n = len(source)
    dim = len(source[0].coordinates)
    if set_size != n:
        raise ValueError("set_size must equal the source vector count")
    if dimension != dim:
        raise ValueError("dimension must equal the source vector dimension")
    return _source_points(source)


def _replay_pair(
    pair: OrderedDifferencePair,
    *,
    points: tuple[tuple[int, ...], ...],
    dimension: int,
    seen_pairs: set[tuple[int, int]],
) -> tuple[int, ...]:
    n = len(points)
    if not (0 <= pair.minuend_index < n and 0 <= pair.subtrahend_index < n):
        raise ValueError("pair indexes must lie in the source set")
    key = (pair.minuend_index, pair.subtrahend_index)
    if key in seen_pairs:
        raise ValueError("ordered pairs must be unique")
    seen_pairs.add(key)
    return tuple(
        points[pair.minuend_index][k] - points[pair.subtrahend_index][k]
        for k in range(dimension)
    )


def _require_replayed_classes(
    classes: tuple[OrderedDifferenceClass, ...],
    *,
    points: tuple[tuple[int, ...], ...],
    dimension: int,
) -> None:
    n = len(points)
    expected_pairs = {(i, j) for i in range(n) for j in range(n) if i != j}
    seen_pairs: set[tuple[int, int]] = set()
    numeric_diffs: list[tuple[int, ...]] = []
    for cls in classes:
        if len(cls.difference.coordinates) != dimension:
            raise ValueError("difference dimension must match the source")
        diff = tuple(parse_canonical_integer(c) for c in cls.difference.coordinates)
        numeric_diffs.append(diff)
        for pair in cls.pairs:
            if (
                _replay_pair(
                    pair,
                    points=points,
                    dimension=dimension,
                    seen_pairs=seen_pairs,
                )
                != diff
            ):
                raise ValueError(
                    "difference must equal source[minuend] - source[subtrahend]",
                )
    if seen_pairs != expected_pairs:
        raise ValueError(
            "classes must cover every ordered pair of distinct source vectors",
        )
    if len(numeric_diffs) != len(set(numeric_diffs)):
        raise ValueError("difference classes must be unique")
    if numeric_diffs != sorted(numeric_diffs):
        raise ValueError("difference classes must be sorted")


def _require_repeated_decision(
    classes: tuple[OrderedDifferenceClass, ...],
    *,
    ordered_pair_count: int,
    support_size: int,
    set_size: int,
    max_multiplicity: int,
    has_repeated_difference: bool,
    first_repeated_difference: tuple[str, ...] | None,
) -> None:
    if ordered_pair_count != set_size * (set_size - 1):
        raise ValueError("ordered_pair_count must equal set_size * (set_size - 1)")
    if support_size != len(classes):
        raise ValueError("support_size must equal the number of difference classes")
    max_mult = max((len(cls.pairs) for cls in classes), default=0)
    if max_multiplicity != max_mult:
        raise ValueError("max_multiplicity must equal the largest class size")
    first_repeated = next(
        (cls.difference.coordinates for cls in classes if len(cls.pairs) > 1),
        None,
    )
    if has_repeated_difference != (max_mult > 1):
        raise ValueError("has_repeated_difference must equal max_multiplicity > 1")
    if first_repeated_difference != first_repeated:
        raise ValueError(
            "first_repeated_difference must be the first class of multiplicity 2+",
        )


class OrderedDifferenceProfileResult(StrictModel):
    """Complete ordered-difference profile of a bounded integer-vector set.

    The result retains the source ``IntegerVectorSet`` so every class can be
    replayed as ``vectors[minuend] - vectors[subtrahend]`` without the request.
    """

    vectors: IntegerVectorSet
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
        points = _require_source_shape(
            self.vectors.vectors,
            dimension=self.dimension,
            set_size=self.set_size,
        )
        _require_replayed_classes(
            self.classes,
            points=points,
            dimension=self.dimension,
        )
        _require_repeated_decision(
            self.classes,
            ordered_pair_count=self.ordered_pair_count,
            support_size=self.support_size,
            set_size=self.set_size,
            max_multiplicity=self.max_multiplicity,
            has_repeated_difference=self.has_repeated_difference,
            first_repeated_difference=self.first_repeated_difference,
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
