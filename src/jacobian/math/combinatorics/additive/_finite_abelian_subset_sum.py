"""Exact subset-sum profiles for finite abelian groups presented by invariant factors."""

from __future__ import annotations

from itertools import product
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError

MAX_FINITE_ABELIAN_GROUP_ORDER = 4_096
MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS = 64
MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS = 1_000_000
MAX_FINITE_ABELIAN_COORDINATE = (1 << 53) - 1
MAX_FINITE_ABELIAN_RANK = 6

# multiplicity bits for 2**64 is 65 bits, digits ~20. For safety allow up to 2**64.
MAX_FINITE_ABELIAN_MULTIPLICITY_BITS = 65
MAX_FINITE_ABELIAN_MULTIPLICITY_DIGITS = (
    MAX_FINITE_ABELIAN_MULTIPLICITY_BITS * 30_103 // 100_000 + 1
)

FiniteAbelianCoordinate = Annotated[
    int,
    Field(
        ge=-MAX_FINITE_ABELIAN_COORDINATE, le=MAX_FINITE_ABELIAN_COORDINATE, strict=True
    ),
]
CanonicalFiniteAbelianCoordinate = Annotated[
    int,
    Field(ge=0, le=MAX_FINITE_ABELIAN_COORDINATE, strict=True),
]
FiniteAbelianMultiplicity = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=MAX_FINITE_ABELIAN_MULTIPLICITY_DIGITS,
        strict=True,
    ),
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


def _validate_invariant_factors(invariant_factors: tuple[int, ...]) -> int:
    if not invariant_factors:
        raise _validation_error(
            "finite_abelian_invariant_factors_empty",
            "invariant factors must be non-empty for finite abelian subset-sum profile",
        )
    if len(invariant_factors) > MAX_FINITE_ABELIAN_RANK:
        raise _validation_error(
            "finite_abelian_rank_exceeds_bound",
            f"finite abelian group rank exceeds the {MAX_FINITE_ABELIAN_RANK}-factor bound",
        )
    if any(f < 2 for f in invariant_factors):
        raise _validation_error(
            "finite_abelian_factor_not_finite",
            "invariant factors must be integers >=2",
        )
    if any(
        invariant_factors[i + 1] % invariant_factors[i] != 0
        for i in range(len(invariant_factors) - 1)
    ):
        raise _validation_error(
            "finite_abelian_factor_divisibility",
            "invariant factors must satisfy d_i | d_{i+1}",
        )
    order = 1
    for f in invariant_factors:
        order *= f
        if order > MAX_FINITE_ABELIAN_GROUP_ORDER:
            raise _validation_error(
                "finite_abelian_group_order_exceeds_bound",
                f"finite abelian group order exceeds the {MAX_FINITE_ABELIAN_GROUP_ORDER}-element bound",
            )
    return order


def _maximum_count_digits(item_count: int) -> int:
    return item_count * 30_103 // 100_000 + 1


class FiniteAbelianSubsetSumEntry(StrictModel):
    """One group element and its exact indexed-subset multiplicity."""

    element: tuple[CanonicalFiniteAbelianCoordinate, ...]
    multiplicity: FiniteAbelianMultiplicity

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        if parse_canonical_integer(self.multiplicity) < 0:
            raise _validation_error(
                "finite_abelian_multiplicity_negative",
                "multiplicity must be non-negative",
            )
        return self


class FiniteAbelianSubsetSumProfileRequest(StrictModel):
    """Complete indexed-subset profile in a finite abelian group.

    The group is Z/d1 x ... x Z/dr with invariant factors d_i. Source elements
    are tuples matching the rank, taken modulo the invariant factors. The empty
    subset contributes 1 to the zero element.
    """

    invariant_factors: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_FINITE_ABELIAN_RANK,
        description=(
            "Invariant factors d_i with d_i >=2 and d_i | d_{i+1}. Product order "
            f"at most {MAX_FINITE_ABELIAN_GROUP_ORDER}."
        ),
    )
    source: tuple[tuple[FiniteAbelianCoordinate, ...], ...] = Field(
        default=(),
        max_length=MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS,
        description=(
            "Indexed sequence of group elements as integer coordinate tuples. "
            f"At most {MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS} positions; each "
            "coordinate is reduced modulo the invariant factor."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        return value

    @model_validator(mode="after")
    def require_valid(self) -> Self:
        order = _validate_invariant_factors(self.invariant_factors)
        rank = len(self.invariant_factors)
        for idx, element in enumerate(self.source):
            if len(element) != rank:
                raise _validation_error(
                    "finite_abelian_source_rank",
                    f"source element at index {idx} must have rank {rank}",
                )
            for coord in element:
                if (
                    not -MAX_FINITE_ABELIAN_COORDINATE
                    <= coord
                    <= MAX_FINITE_ABELIAN_COORDINATE
                ):
                    raise _validation_error(
                        "finite_abelian_coordinate_bound",
                        "source coordinate exceeds the 2**53 bound",
                    )
        # DP bound checked in admission, but also ensure source length
        if len(self.source) > MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS:
            raise _validation_error(
                "finite_abelian_item_count",
                f"source length exceeds the {MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS}-item bound",
            )
        dp_cells = len(self.source) * order
        if dp_cells > MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS:
            raise _validation_error(
                "finite_abelian_dp_cells",
                f"finite abelian DP {len(self.source)}*{order}={dp_cells} exceeds the {MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS:,}-cell bound",
            )
        return self


class FiniteAbelianSubsetSumProfileResult(StrictModel):
    """Complete multiplicity profile bound to its presentation and source."""

    invariant_factors: tuple[int, ...] = Field(
        min_length=1, max_length=MAX_FINITE_ABELIAN_RANK
    )
    source: tuple[tuple[FiniteAbelianCoordinate, ...], ...] = Field(
        max_length=MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS
    )
    entries: tuple[FiniteAbelianSubsetSumEntry, ...] = Field(
        min_length=1, max_length=MAX_FINITE_ABELIAN_GROUP_ORDER
    )
    support_size: int = Field(ge=0, le=MAX_FINITE_ABELIAN_GROUP_ORDER)
    covers_group: bool
    total_subsets: FiniteAbelianMultiplicity

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        value = canonicalize_json_containers(value)
        return value

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        order = _validate_invariant_factors(self.invariant_factors)
        if len(self.entries) != order:
            raise _validation_error(
                "finite_abelian_result_order",
                "entries must contain exactly |G| rows",
            )
        # Check that entries are sorted and unique
        elements = tuple(entry.element for entry in self.entries)
        if elements != tuple(sorted(elements)):
            raise _validation_error(
                "finite_abelian_entries_sorted",
                "entries must be sorted lexicographically by element",
            )
        if len(set(elements)) != len(elements):
            raise _validation_error(
                "finite_abelian_entries_unique",
                "entries must have unique elements",
            )
        # Check each element canonical
        rank = len(self.invariant_factors)
        for element in elements:
            if len(element) != rank:
                raise _validation_error(
                    "finite_abelian_entry_rank",
                    "entry element rank must match invariant factors",
                )
            for coord, modulus in zip(element, self.invariant_factors, strict=True):
                if not 0 <= coord < modulus:
                    raise _validation_error(
                        "finite_abelian_entry_canonical",
                        "entry element must be canonical 0 <= coord < d_i",
                    )
        if self.support_size != sum(
            1
            for entry in self.entries
            if parse_canonical_integer(entry.multiplicity) > 0
        ):
            raise _validation_error(
                "finite_abelian_support_size",
                "support_size must equal number of entries with positive multiplicity",
            )
        if self.covers_group != (self.support_size == order):
            raise _validation_error(
                "finite_abelian_covers_group",
                "covers_group must be true iff support_size == |G|",
            )
        total = sum(
            parse_canonical_integer(entry.multiplicity) for entry in self.entries
        )
        if parse_canonical_integer(self.total_subsets) != total:
            raise _validation_error(
                "finite_abelian_total_subsets",
                "total_subsets must equal sum of multiplicities",
            )
        # total must be 2^k
        expected_total = (
            1 << len(self.source) if len(self.source) < 63 else pow(2, len(self.source))
        )
        # For larger k we compute via pow
        if len(self.source) >= 63:
            expected_total = pow(2, len(self.source))
        if parse_canonical_integer(self.total_subsets) != expected_total:
            raise _validation_error(
                "finite_abelian_total_subsets_power",
                "total_subsets must equal 2^{len(source)}",
            )
        return self

    @classmethod
    def _from_kernel(
        cls,
        invariant_factors: tuple[int, ...],
        source: tuple[tuple[int, ...], ...],
        entries: tuple[FiniteAbelianSubsetSumEntry, ...],
        support_size: int,
        covers_group: bool,
        total_subsets: str,
    ) -> Self:
        return cls.model_construct(
            invariant_factors=invariant_factors,
            source=source,
            entries=entries,
            support_size=support_size,
            covers_group=covers_group,
            total_subsets=total_subsets,
        )


def _all_group_elements(invariant_factors: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Return all canonical elements sorted lexicographically."""
    ranges = [range(d) for d in invariant_factors]
    elements = list(product(*ranges))
    # product already yields lexicographic with last varying fastest, but we need sorted
    return sorted(elements)


def _reduce_element(
    element: tuple[int, ...], invariant_factors: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        coord % modulus
        for coord, modulus in zip(element, invariant_factors, strict=True)
    )


def _add_elements(
    a: tuple[int, ...], b: tuple[int, ...], invariant_factors: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        (x + y) % modulus for x, y, modulus in zip(a, b, invariant_factors, strict=True)
    )


def _admit_finite_abelian_subset_sum(
    invariant_factors: tuple[int, ...],
    source: tuple[tuple[int, ...], ...],
) -> None:
    try:
        order = _validate_invariant_factors(invariant_factors)
    except PydanticCustomError as error:
        raise OperationDomainValidationError(
            location=("invariant_factors",),
            code=f"additive_combinatorics.{error.type}",
            message=str(error),
        ) from error
    rank = len(invariant_factors)
    if len(source) > MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS:
        raise OperationDomainValidationError(
            location=("source",),
            code="additive_combinatorics.finite_abelian_subset_sum.item_count",
            message=f"source length exceeds the {MAX_FINITE_ABELIAN_SUBSET_SUM_ITEMS}-item bound",
        )
    for idx, element in enumerate(source):
        if len(element) != rank:
            raise OperationDomainValidationError(
                location=("source", idx),
                code="additive_combinatorics.finite_abelian_subset_sum.rank_mismatch",
                message=f"source element at index {idx} must have rank {rank}",
            )
        for coord in element:
            if (
                not -MAX_FINITE_ABELIAN_COORDINATE
                <= coord
                <= MAX_FINITE_ABELIAN_COORDINATE
            ):
                raise OperationDomainValidationError(
                    location=("source", idx),
                    code="additive_combinatorics.finite_abelian_subset_sum.coordinate_bound",
                    message="source coordinate exceeds the 2**53 bound",
                )
    dp_cells = len(source) * order
    if dp_cells > MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS:
        raise OperationDomainValidationError(
            location=("source",),
            code="additive_combinatorics.finite_abelian_subset_sum.dp_bound",
            message=f"DP {len(source)}*{order}={dp_cells} exceeds the {MAX_FINITE_ABELIAN_SUBSET_SUM_DP_CELLS:,}-cell bound",
        )


def finite_abelian_subset_sum_profile(
    invariant_factors: tuple[int, ...],
    source: tuple[tuple[int, ...], ...],
) -> FiniteAbelianSubsetSumProfileResult:
    """Return the complete multiplicity profile for a finite abelian group.

    Counts include the empty subset at the zero element. Uses dense DP
    ``c_new(g)=c_old(g)+c_old(g-a_i)`` over the product group.
    """
    # Normalize source to tuples (ensure immutability)
    source = tuple(tuple(int(c) for c in elem) for elem in source)
    invariant_factors = tuple(int(d) for d in invariant_factors)
    _admit_finite_abelian_subset_sum(invariant_factors, source)
    # Reduce source canonically
    canonical_source = tuple(
        _reduce_element(elem, invariant_factors) for elem in source
    )
    elements = _all_group_elements(invariant_factors)
    element_to_index = {elem: idx for idx, elem in enumerate(elements)}
    counts = [0] * len(elements)
    zero = tuple(0 for _ in invariant_factors)
    counts[element_to_index[zero]] = 1  # empty subset
    for a in canonical_source:
        # Use snapshot of current counts
        old_counts = counts.copy()
        for g, mult in enumerate(old_counts):
            if mult == 0:
                continue
            g_elem = elements[g]
            target = _add_elements(g_elem, a, invariant_factors)
            t_idx = element_to_index[target]
            counts[t_idx] += mult
        # also need to handle case where old_counts already had empty? The above includes it via loop.
        # But we also need to ensure counts for new element directly? The loop already adds old_counts[g] to target,
        # which includes the empty subset contributions.
        # No separate handling needed; the recurrence counts[g] stays, and counts[target] gets added.

    entries = tuple(
        FiniteAbelianSubsetSumEntry(
            element=elem,
            multiplicity=format_canonical_integer(count),
        )
        for elem, count in zip(elements, counts, strict=True)
    )
    support_size = sum(1 for c in counts if c > 0)
    covers_group = support_size == len(elements)
    total_subsets = format_canonical_integer(
        1 << len(source) if len(source) < 1024 else pow(2, len(source))
    )
    # Use pow for large
    if len(source) >= 1024:
        total_subsets = format_canonical_integer(pow(2, len(source)))
    elif len(source) >= 64:
        # For 64..1023, pow already works but 1<<n may overflow Python int? Python handles big ints.
        total_subsets = format_canonical_integer(pow(2, len(source)))
    return FiniteAbelianSubsetSumProfileResult._from_kernel(
        invariant_factors=invariant_factors,
        source=canonical_source,
        entries=entries,
        support_size=support_size,
        covers_group=covers_group,
        total_subsets=total_subsets,
    )
