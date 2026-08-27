"""Typed wire contracts for additive combinatorics operations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import (
    CanonicalLimits,
    format_canonical_integer,
    parse_canonical_integer,
    strict_json_object_size,
)
from jacobian.math.additive_combinatorics import _multiset_sum
from jacobian.math.additive_combinatorics._subset_sum_profile import (
    MAX_SUBSET_SUM_DP_TRANSITIONS,
    MAX_SUBSET_SUM_PROFILE_RESULT_BYTES,
)
from jacobian.math.additive_combinatorics.values import (
    MAX_SUBSET_SUM_ITEMS,
    MAX_SUBSET_SUM_PROFILE_ENTRIES,
    IndexedIntegerSequence,
    indexed_sequence_item_ceiling,
)
from jacobian.math.finite_sets._models import FiniteIntegerSet

# This conservative materialized-axis cap bounds source parsing and binomial
# preflight. Operation-specific work and result bounds impose the sharper
# execution envelope; binary Cartesian operations retain their pair cap below.
_MAX_SET_SIZE = 4096
_MAX_CARTESIAN_PAIR_COUNT = 256 * 256
_MAX_RESULT_SIZE = _MAX_SET_SIZE * _MAX_SET_SIZE
_MAX_DIMENSION = 8
_MAX_COORDINATE_DIGITS = 6
_MAX_MULTISET_SUM_ELEMENT_DIGITS = _multiset_sum.MAX_ELEMENT_DIGITS
_MAX_MULTISET_SUM_ARITY = _multiset_sum.MAX_ARITY
_MAX_MULTISET_SUM_ENUMERATION_WORK = _multiset_sum.MAX_ENUMERATION_WORK
_MAX_MULTISET_SUM_INTEGER_LENGTH = _multiset_sum.MAX_INTEGER_LENGTH
_MAX_MULTISET_SUM_RESULT_DIGITS = _multiset_sum.MAX_RESULT_DIGITS
_MAX_MULTISET_SUM_SUPPORT_SIZE = _multiset_sum.MAX_SUPPORT_SIZE

# ``direct_sum_predicate`` returns a complete partition of Z_n: every residue
# occurs exactly once in either ``representatives`` or ``missing``.  Collisions
# are an additional, distinct-residue diagnostic.  Reserve the real canonical
# transport budget before enumerating the missing set, rather than allowing a
# valid computation to fail only when dispatch serializes its result.
_MAX_DIRECT_SUM_RESULT_BYTES = CanonicalLimits().max_output_bytes

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


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


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

CanonicalMultisetSumBound = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|-?[1-9][0-9]*)$",
        max_length=_MAX_MULTISET_SUM_INTEGER_LENGTH,
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
        raise _validation_error(
            "_require_bounded_coordinate",
            f"{label} exceeds the {_MAX_COORDINATE_DIGITS}-digit coordinate bound",
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
                raise _validation_error(
                    "require_uniform_distinct_bounded",
                    "all vectors must share the same dimension",
                )
            if not 1 <= dimension <= _MAX_DIMENSION:
                raise _validation_error(
                    "require_uniform_distinct_bounded",
                    f"vector dimension must be between 1 and {_MAX_DIMENSION}",
                )
            if values in seen:
                raise _validation_error(
                    "require_uniform_distinct_bounded", "vectors must be unique"
                )
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
        raise _validation_error(
            "_check_totals", "total ordered pairs must match sum of multiplicities"
        )
    if total_ordered_pairs != set_size * (set_size - 1):
        raise _validation_error(
            "_check_totals", "total_ordered_pairs must equal set_size*(set_size-1)"
        )
    if support_size != len(entries):
        raise _validation_error(
            "_check_totals", "support_size must equal the number of entries"
        )


def _check_max_and_repeated(
    entries: tuple[OrderedDifferenceEntry, ...],
    max_multiplicity: int,
    has_repeated_difference: bool,
    first_collision: OrderedDifferencePair | None,
) -> None:
    if not entries:
        if max_multiplicity != 0:
            raise _validation_error(
                "_check_max_and_repeated",
                "max_multiplicity must be 0 when entries is empty",
            )
    elif max_multiplicity != max(e.multiplicity for e in entries):
        raise _validation_error(
            "_check_max_and_repeated",
            "max_multiplicity must be the maximum entry multiplicity",
        )
    expected_repeated = (max_multiplicity > 1) if entries else False
    if has_repeated_difference != expected_repeated:
        raise _validation_error(
            "_check_max_and_repeated",
            "has_repeated_difference must match max_multiplicity > 1",
        )
    if has_repeated_difference and first_collision is None:
        raise _validation_error(
            "_check_max_and_repeated",
            "first_collision must be present when has_repeated_difference",
        )
    if not has_repeated_difference and first_collision is not None:
        raise _validation_error(
            "_check_max_and_repeated",
            "first_collision must be null when has_repeated_difference is false",
        )


def _check_entries_sorted(entries: tuple[OrderedDifferenceEntry, ...]) -> None:
    diffs = [entry.difference.as_int_tuple() for entry in entries]
    if diffs != sorted(diffs):
        raise _validation_error(
            "_check_entries_sorted", "entries must be sorted by difference"
        )
    if len(set(diffs)) != len(diffs):
        raise _validation_error(
            "_check_entries_sorted", "entries differences must be unique"
        )


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
            raise _validation_error(
                "_check_entry_pairs",
                "entry difference dimension must match result dimension",
            )
        if difference == zero:
            raise _validation_error(
                "_check_entry_pairs", "entry difference must be nonzero"
            )
        previous: tuple[int, int] | None = None
        for pair in entry.pairs:
            key = (pair.left_index, pair.right_index)
            if previous is not None and key <= previous:
                raise _validation_error(
                    "_check_entry_pairs",
                    "entry pairs must be sorted and unique in lexicographic order",
                )
            previous = key
            if pair.left_index >= set_size or pair.right_index >= set_size:
                raise _validation_error(
                    "_check_entry_pairs", "pair indices must be less than set_size"
                )
            if vectors:
                expected = tuple(
                    vectors[pair.left_index].as_int_tuple()[k]
                    - vectors[pair.right_index].as_int_tuple()[k]
                    for k in range(dimension)
                )
                if expected != difference:
                    raise _validation_error(
                        "_check_entry_pairs", "pair difference must match vectors"
                    )


def _check_all_pairs_exactly_once(
    entries: tuple[OrderedDifferenceEntry, ...],
    set_size: int,
) -> None:
    seen: set[tuple[int, int]] = set()
    for entry in entries:
        for pair in entry.pairs:
            key = (pair.left_index, pair.right_index)
            if key in seen:
                raise _validation_error(
                    "_check_all_pairs_exactly_once",
                    f"ordered pair {key} appears more than once",
                )
            seen.add(key)
    expected: set[tuple[int, int]] = {
        (i, j) for i in range(set_size) for j in range(set_size) if i != j
    }
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise _validation_error(
            "_check_all_pairs_exactly_once",
            f"entries must contain every ordered pair exactly once; "
            f"missing {sorted(missing)[:5]}, extra {sorted(extra)[:5]}",
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
            raise _validation_error(
                "_check_first_collision",
                "has_repeated_difference requires a repeated difference entry",
            )
        if first_collision != expected_entry.pairs[0]:
            raise _validation_error(
                "_check_first_collision",
                "first_collision must be the designated pair of the first "
                "repeated-difference entry",
            )
    elif not entries and first_collision is not None:
        raise _validation_error(
            "_check_first_collision",
            "first_collision must be null when entries is empty",
        )


def _require_bounded_cartesian_product(
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> None:
    pair_count = len(left.elements) * len(right.elements)
    if pair_count > _MAX_CARTESIAN_PAIR_COUNT:
        raise _validation_error(
            "_require_bounded_cartesian_product",
            f"Cartesian product has {pair_count} pairs, exceeding the "
            f"{_MAX_CARTESIAN_PAIR_COUNT}-pair bound",
        )


def _decimal_digit_sum_through(value: int) -> int:
    """Return the total decimal digit count of the integers in ``[0, value)``."""

    if value <= 0:
        return 0
    total = 0
    lower = 1
    digits = 1
    while lower < value:
        upper = min(value, lower * 10)
        total += (upper - lower) * digits
        lower = upper
        digits += 1
    return total + 1  # The residue 0 has one decimal digit.


def _direct_sum_predicate_result_upper_bound(
    modulus: int,
    source_pair_count: int,
) -> int:
    """Bound the canonical JSON result for one admitted direct-sum request.

    The representatives and missing lists partition the residue classes, so
    their combined decimal text is exact. A collision needs two distinct
    source pairs, hence at most ``source_pair_count // 2`` distinct collision
    residues can be reported. For those optional entries, charging the widest
    residue text is conservative.
    """

    partition_value_bytes = (
        _decimal_digit_sum_through(modulus)
        + 2 * modulus  # JSON string quotes
        + modulus  # at most one comma per partition entry across two arrays
        + 4  # the two array delimiters
    )
    collision_count = min(modulus, source_pair_count // 2)
    collision_digit_bound = len(str(modulus - 1))
    collision_value_bytes = (
        2 if collision_count == 0 else collision_count * (collision_digit_bound + 3) + 1
    )
    return (
        strict_json_object_size(
            (
                ("holds", len("false")),
                ("modulus", len(str(modulus))),
                ("representatives", 0),
                ("collisions", collision_value_bytes),
                ("missing", 0),
            )
        )
        + partition_value_bytes
    )


def _require_direct_sum_result_transport_bound(
    modulus: int,
    left: FiniteIntegerSet,
    right: FiniteIntegerSet,
) -> None:
    source_pair_count = len(left.elements) * len(right.elements)
    predicted = _direct_sum_predicate_result_upper_bound(
        modulus,
        source_pair_count,
    )
    if predicted > _MAX_DIRECT_SUM_RESULT_BYTES:
        raise _validation_error(
            "direct_sum_result_transport_exceeded",
            "direct-sum diagnostics would use up to "
            f"{predicted:,} canonical JSON bytes, exceeding the "
            f"{_MAX_DIRECT_SUM_RESULT_BYTES:,}-byte output limit; reduce the "
            "modulus or partition the residue classes",
        )


class FiniteCyclicGroup(StrictModel):
    """The cyclic group ``Z_n`` carrying a direct-sum/tiling predicate."""

    modulus: int = Field(gt=1, le=_MAX_RESULT_SIZE)

    @model_validator(mode="after")
    def require_valid_modulus(self) -> Self:
        if self.modulus < 2:
            raise _validation_error(
                "require_valid_modulus", "cyclic group modulus must be at least 2"
            )
        return self


# ---------------------------------------------------------------------------
# Representation profile
# ---------------------------------------------------------------------------


class RepresentationProfileRequest(StrictModel):
    """Compute ``r_{A+B}(x)`` for every sum ``x`` of two finite integer sets.

    The complete Cartesian product contains at most 65,536 source pairs.
    """

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
            raise _validation_error(
                "require_canonical_entries",
                "representation profile sums must be sorted and unique",
            )
        if any(entry.multiplicity <= 0 for entry in self.entries):
            raise _validation_error(
                "require_canonical_entries",
                "representation multiplicities must be positive",
            )
        return self


# ---------------------------------------------------------------------------
# Fixed-arity unordered multiset-sum representation profile
# ---------------------------------------------------------------------------


class MultisetSumWindow(StrictModel):
    """One closed integer interval restricting a complete sum profile."""

    lower: CanonicalMultisetSumBound = Field(
        description=(
            "Canonical lower endpoint of the closed sum window, carrying at "
            f"most {_MAX_MULTISET_SUM_RESULT_DIGITS} digits plus an optional sign."
        ),
        examples=["0"],
    )
    upper: CanonicalMultisetSumBound = Field(
        description=(
            "Canonical upper endpoint of the closed sum window, carrying at "
            f"most {_MAX_MULTISET_SUM_RESULT_DIGITS} digits plus an optional sign; "
            "it must be at least lower."
        ),
        examples=["10"],
    )

    @model_validator(mode="after")
    def require_nondecreasing_endpoints(self) -> Self:
        if any(
            len(endpoint.lstrip("-")) > _MAX_MULTISET_SUM_RESULT_DIGITS
            for endpoint in (self.lower, self.upper)
        ):
            raise _validation_error(
                "require_nondecreasing_endpoints",
                "sum window endpoints must carry at most "
                f"{_MAX_MULTISET_SUM_RESULT_DIGITS} digits",
            )
        lower, upper = self.as_integer_bounds()
        if lower > upper:
            raise _validation_error(
                "require_nondecreasing_endpoints",
                "sum window lower endpoint must not exceed upper",
            )
        return self

    def as_integer_bounds(self) -> tuple[int, int]:
        return (
            parse_canonical_integer(self.lower),
            parse_canonical_integer(self.upper),
        )


def _multiset_sum_source_values(source: FiniteIntegerSet) -> tuple[int, ...]:
    for element in source.elements:
        if len(element.lstrip("-")) > _MAX_MULTISET_SUM_ELEMENT_DIGITS:
            raise _validation_error(
                "_multiset_sum_source_values",
                "multiset-sum source elements must carry at most "
                f"{_MAX_MULTISET_SUM_ELEMENT_DIGITS} digits",
            )
    values = tuple(parse_canonical_integer(element) for element in source.elements)
    if values != tuple(sorted(values)):
        raise _validation_error(
            "_multiset_sum_source_values",
            "multiset-sum source elements must be in strictly increasing numeric order",
        )
    return values


_MULTISET_SUM_SOURCE_DESCRIPTION = (
    "A materialized finite set of distinct canonical integers in strictly "
    "increasing numeric order; each element carries at "
    f"most {_MAX_MULTISET_SUM_ELEMENT_DIGITS} digits."
)


class MultisetSumRepresentationProfileRequest(StrictModel):
    """Compute one complete fixed-arity unordered multiset-sum profile.

    ``source`` is a materialized finite set in strictly increasing numeric order.
    The operation inspects every nondecreasing source-index tuple of length
    ``arity``. With a window, completeness and missing-row-as-zero semantics are
    restricted to that closed interval; without one, every attainable sum is
    returned. Admission bounds candidate enumeration, bigint growth, and the
    worst-case exact support before execution.
    """

    source: FiniteIntegerSet = Field(
        description=_MULTISET_SUM_SOURCE_DESCRIPTION,
        examples=[{"elements": ["0", "1", "2"]}],
    )
    arity: int = Field(
        ge=0,
        le=_MAX_MULTISET_SUM_ARITY,
        description=(
            f"Nonnegative multiset arity carrying at most "
            f"{_multiset_sum.MAX_ARITY_DIGITS} decimal digits. Arity zero has "
            "one empty multiset with sum zero, including when the source is "
            "empty; admission derives the accepted envelope from candidate "
            "work and predicted support rather than from this magnitude."
        ),
        examples=[2],
    )
    window: MultisetSumWindow | None = Field(
        default=None,
        description=(
            "Optional closed sum interval. Null requests the complete profile; "
            "a window returns every and only attainable sum inside that interval."
        ),
        examples=[None, {"lower": "0", "upper": "10"}],
    )

    @model_validator(mode="after")
    def require_bounded_complete_enumeration(self) -> Self:
        values = _multiset_sum_source_values(self.source)
        candidate_count = _multiset_sum.candidate_count(len(values), self.arity)
        bounds = self.window.as_integer_bounds() if self.window is not None else None
        work = _multiset_sum.enumeration_work(
            values, self.arity, bounds, candidate_count
        )
        if work > _MAX_MULTISET_SUM_ENUMERATION_WORK:
            raise _validation_error(
                "require_bounded_complete_enumeration",
                f"multiset-sum enumeration requires {work} coordinate steps, "
                f"exceeding the {_MAX_MULTISET_SUM_ENUMERATION_WORK}-step bound",
            )
        support_bound = _multiset_sum.support_bound(
            values, self.arity, bounds, candidate_count
        )
        if support_bound > _MAX_MULTISET_SUM_SUPPORT_SIZE:
            raise _validation_error(
                "require_bounded_complete_enumeration",
                f"multiset-sum profile may contain {support_bound} rows, exceeding "
                f"the {_MAX_MULTISET_SUM_SUPPORT_SIZE}-row result bound; supply a "
                "narrower closed sum window",
            )
        return self


class MultisetSumRepresentationProfileResult(StrictModel):
    """Source-bound exact multiplicities for one complete sum scope.

    For each row ``s -> m``, ``m`` is the number of nondecreasing source-index
    tuples of the declared arity summing to ``s``. Deserialization validates
    only the bounded canonical shape; the owner-local verifier checks an
    independently supplied complete claim under the request envelope.
    """

    source: FiniteIntegerSet = Field(description=_MULTISET_SUM_SOURCE_DESCRIPTION)
    arity: int = Field(ge=0, le=_MAX_MULTISET_SUM_ARITY)
    window: MultisetSumWindow | None = None
    entries: tuple[RepresentationProfileEntry, ...] = Field(
        default=(), max_length=_MAX_MULTISET_SUM_SUPPORT_SIZE
    )

    @model_validator(mode="after")
    def require_canonical_entries(self) -> Self:
        sums = tuple(entry.sum for entry in self.entries)
        if sums != _sorted_canonical_integers(sums):
            raise _validation_error(
                "multiset_sum_profile_entries",
                "multiset-sum entries must be sorted and unique",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: MultisetSumRepresentationProfileRequest,
        entries: tuple[RepresentationProfileEntry, ...],
    ) -> Self:
        return cls(
            source=request.source,
            arity=request.arity,
            window=request.window,
            entries=entries,
        )


# ---------------------------------------------------------------------------
# Complete indexed subset-sum profile
# ---------------------------------------------------------------------------


class SubsetSumProfileRequest(StrictModel):
    """Compute every indexed subset-sum multiplicity, including the empty set.

    Admission is result-sensitive: it bounds the exact sum span, the number of
    source-selection vectors, sparse-DP transitions, multiplicity digits, and
    worst-case serialized profile before the dynamic program begins.
    """

    source: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_SUBSET_SUM_ITEMS)),
    ] = Field(
        description=(
            "The ordered indexed integer sequence, at most 4,095 items. Each "
            "position is selectable at most once; repeated values and zeros "
            "remain distinct positions. Before execution, S=min(2^n, "
            "positive_sum-negative_sum+1, product(m_v+1) over distinct "
            f"nonzero values v) must fit {MAX_SUBSET_SUM_PROFILE_ENTRIES:,} "
            f"rows, 4*n*S must not exceed {MAX_SUBSET_SUM_DP_TRANSITIONS:,} "
            "dictionary transitions during construction, "
            "and the conservative serialized-result estimate must not exceed "
            f"{MAX_SUBSET_SUM_PROFILE_RESULT_BYTES:,} bytes."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: object) -> object:
        """Reject oversized raw sources before nested integer parsing.

        Running a before validator moves field validation into Python
        mode, where decoded JSON arrays no longer coerce to the declared
        tuple shapes; normalize the source value list to a tuple on a
        copied path so JSON invocation keeps working while the stored
        sequence stays canonical. Container canonicalization preserves
        element count and order, so the raw item-count bound is enforced
        after it against either accepted container shape.
        """

        value = canonicalize_json_containers(value)

        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_source = prepared.get("source")
        if isinstance(raw_source, Mapping):
            source = dict(raw_source)
            items = source.get("items")
            if isinstance(items, (list, tuple)):
                source["items"] = tuple(items)
                prepared["source"] = source
                if len(items) > MAX_SUBSET_SUM_ITEMS:
                    raise _validation_error(
                        "bound_raw_source",
                        "subset-sum profile source exceeds the "
                        f"{MAX_SUBSET_SUM_ITEMS:,}-item profile bound",
                    )
            else:
                prepared["source"] = source
        return prepared

    @model_validator(mode="after")
    def require_bounded_source_shape(self) -> Self:
        """Keep the declared request container limit for typed callers too."""

        if len(self.source.items) > MAX_SUBSET_SUM_ITEMS:
            raise _validation_error(
                "bound_raw_source",
                "subset-sum profile source exceeds the "
                f"{MAX_SUBSET_SUM_ITEMS:,}-item profile bound",
            )
        return self


# ---------------------------------------------------------------------------
# Additive energy
# ---------------------------------------------------------------------------


class AdditiveEnergyRequest(StrictModel):
    """Compute the additive energy ``E(A, B) = sum_x r_{A+B}(x)^2``.

    The complete Cartesian product contains at most 65,536 source pairs.
    """

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
            raise _validation_error(
                "require_canonical_decomposition",
                "additive energy sums must be sorted and unique",
            )
        if any(entry.multiplicity <= 0 for entry in self.decomposition):
            raise _validation_error(
                "require_canonical_decomposition",
                "additive energy multiplicities must be positive",
            )
        if self.energy != sum(entry.multiplicity**2 for entry in self.decomposition):
            raise _validation_error(
                "require_canonical_decomposition",
                "additive energy must equal the sum of squared multiplicities",
            )
        return self

    @classmethod
    def _from_kernel(
        cls, energy: int, decomposition: tuple[RepresentationProfileEntry, ...]
    ) -> Self:
        return cls(energy=energy, decomposition=decomposition)


# ---------------------------------------------------------------------------
# Sumset cardinality
# ---------------------------------------------------------------------------


class SumsetCardinalityRequest(StrictModel):
    """Compute ``|A + B|`` (the support cardinality of ``r_{A+B}``).

    The complete Cartesian product contains at most 65,536 source pairs.
    """

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
            raise _validation_error(
                "require_canonical_support", "sumset support must be sorted and unique"
            )
        if self.cardinality != len(self.support):
            raise _validation_error(
                "require_canonical_support", "cardinality must equal the support length"
            )
        return self

    @classmethod
    def _from_kernel(cls, support: tuple[CanonicalInteger, ...]) -> Self:
        return cls(cardinality=len(support), support=support)


# ---------------------------------------------------------------------------
# Direct sum / tiling predicate in Z_n
# ---------------------------------------------------------------------------


class DirectSumPredicateRequest(StrictModel):
    """Decide whether ``A (\\oplus) B = Z_n`` inside a finite cyclic group.

    The complete Cartesian product contains at most 65,536 source pairs.
    """

    modulus: int = Field(gt=1, le=_MAX_RESULT_SIZE)
    left: FiniteIntegerSet
    right: FiniteIntegerSet

    @model_validator(mode="after")
    def require_bounded_cartesian_product(self) -> Self:
        _require_bounded_cartesian_product(self.left, self.right)
        _require_direct_sum_result_transport_bound(
            self.modulus,
            self.left,
            self.right,
        )
        return self


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
                raise _validation_error(
                    "require_canonical_diagnostics",
                    f"direct-sum {name} values must be sorted and unique",
                )
        return self

    @classmethod
    def _from_kernel(
        cls,
        *,
        holds: bool,
        modulus: int,
        representatives: tuple[CanonicalInteger, ...],
        collisions: tuple[CanonicalInteger, ...],
        missing: tuple[CanonicalInteger, ...],
    ) -> Self:
        return cls(
            holds=holds,
            modulus=modulus,
            representatives=representatives,
            collisions=collisions,
            missing=missing,
        )


__all__ = [
    "AdditiveEnergyRequest",
    "AdditiveEnergyResult",
    "DirectSumPredicateRequest",
    "DirectSumPredicateResult",
    "FiniteCyclicGroup",
    "FiniteIntegerSet",
    "MultisetSumRepresentationProfileRequest",
    "MultisetSumRepresentationProfileResult",
    "MultisetSumWindow",
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
            raise _validation_error(
                "require_canonical", "multiplicity must equal the number of pairs"
            )
        for pair in self.pairs:
            if pair.left_index == pair.right_index:
                raise _validation_error(
                    "require_canonical", "pair indices must be distinct"
                )
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
        # bounded dimension, and nonemptiness.
        if len(self.vectors.vectors) != self.set_size:
            raise _validation_error(
                "require_vectors", "vectors length must equal set_size"
            )
        if len(self.vectors.vectors[0].coordinates) != self.dimension:
            raise _validation_error("require_vectors", "dimension must match vectors")
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
            _check_first_collision(
                self.entries, self.has_repeated_difference, self.first_collision
            )
        elif self.first_collision is not None:
            raise _validation_error(
                "require_entries", "first_collision must be null when entries is empty"
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: OrderedDifferenceProfileRequest,
        *,
        dimension: int,
        total_ordered_pairs: int,
        support_size: int,
        max_multiplicity: int,
        entries: tuple[OrderedDifferenceEntry, ...],
        has_repeated_difference: bool,
        first_collision: OrderedDifferencePair | None,
    ) -> Self:
        return cls(
            vectors=request.vectors,
            dimension=dimension,
            set_size=len(request.vectors.vectors),
            total_ordered_pairs=total_ordered_pairs,
            support_size=support_size,
            max_multiplicity=max_multiplicity,
            entries=entries,
            has_repeated_difference=has_repeated_difference,
            first_collision=first_collision,
        )
