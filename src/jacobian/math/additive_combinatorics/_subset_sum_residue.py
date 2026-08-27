"""Exact modular subset-sum profiles for indexed integer sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Self

from pydantic import Field, StrictBool, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.additive_combinatorics.values import (
    IndexedIntegerSequence,
    IndexSubset,
    indexed_sequence_item_ceiling,
)

# The dense residue recurrence visits exactly n*m cells. Counts never exceed
# 2^n; their bit bound therefore covers every bigint intermediate.
# The modulus ceiling controls the two dense arrays and required m-entry result
# even for an empty source.
MAX_RESIDUE_PROFILE_DP_CELLS = 1_000_000
MAX_RESIDUE_PROFILE_MODULUS = 65_536
MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS = 4_096
MAX_RESIDUE_PROFILE_ITEMS = MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS - 1
MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS = 32_768

# A count-only result may use the broader DP envelope.  Canonical witnesses
# retain at most n indices for each of m residues, so witness-bearing requests
# have their own output-sensitive index-slot budget.
MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS = 250_000
MAX_RESIDUE_PROFILE_RESULT_BYTES = 4 * 1024 * 1024

# 30103/100000 is a strict upper bound for log_10(2).  The derived character
# ceiling covers every count through 2^(MAX_BITS-1) without constructing it.
MAX_RESIDUE_PROFILE_MULTIPLICITY_DIGITS = (
    MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS - 1
) * 30_103 // 100_000 + 1

ResidueMultiplicity = Annotated[
    str,
    StringConstraints(
        pattern=r"^(?:0|[1-9][0-9]*)$",
        max_length=MAX_RESIDUE_PROFILE_MULTIPLICITY_DIGITS,
        strict=True,
    ),
]


def _maximum_count_digits(item_count: int) -> int:
    """Return a conservative decimal-digit bound for ``2**item_count``."""
    return item_count * 30_103 // 100_000 + 1


def _estimated_result_bytes(
    source: IndexedIntegerSequence,
    modulus: int,
    *,
    include_witnesses: bool,
) -> int:
    """Conservatively bound the canonical result before running the DP."""
    item_count = len(source.items)
    source_bytes = sum(len(value) + 3 for value in source.items)
    count_bytes = modulus * (_maximum_count_digits(item_count) + 3)

    witness_bytes = 0
    if include_witnesses:
        index_digits = len(str(max(item_count - 1, 0)))
        # This treats every residue as reachable and every witness as containing
        # every source index.  Model keys, brackets, commas, and ``null`` rows
        # fit within the 24-byte per-row structural allowance.
        witness_bytes = modulus * (24 + item_count * (index_digits + 1))

    # The fixed allowance covers result field names, the nested source object,
    # booleans, modulus digits, brackets, and the outer operation envelope.
    return 1_024 + source_bytes + count_bytes + witness_bytes


def _raw_source_shape(source: object) -> tuple[int, int] | None:
    """Bound a raw indexed source before Pydantic parses its integer strings."""

    if isinstance(source, IndexedIntegerSequence):
        values: list[object] | tuple[object, ...] = source.items
    elif isinstance(source, Mapping):
        raw_values = source.get("items")
        if not isinstance(raw_values, (list, tuple)):
            return None
        values = raw_values
    else:
        return None

    item_count = len(values)
    if item_count + 1 > MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS:
        raise _validation_error(
            "_raw_source_shape",
            "subset multiplicities exceed the 4,096-bit intermediate bound",
        )

    source_bytes = 1_024
    for index, value in enumerate(values):
        if not isinstance(value, str):
            continue
        digit_count = len(value) - value.startswith("-")
        if digit_count > MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS:
            raise _validation_error(
                "_raw_source_shape",
                f"source value at index {index} exceeds the 32,768-digit input bound",
            )
        source_bytes += len(value) + 3
        if source_bytes > MAX_RESIDUE_PROFILE_RESULT_BYTES:
            raise _validation_error(
                "_raw_source_shape",
                "subset-sum residue profile exceeds the 4 MiB result bound",
            )
    return item_count, source_bytes


def _bound_raw_counts(
    counts: object,
    *,
    expected_rows: int | None,
    item_count: int,
    result_bytes: int,
) -> int:
    if not isinstance(counts, (list, tuple)):
        return result_bytes
    if len(counts) > MAX_RESIDUE_PROFILE_MODULUS:
        raise _validation_error(
            "_bound_raw_counts", "residue_counts exceeds the bounded result cardinality"
        )
    if expected_rows is not None and len(counts) != expected_rows:
        raise _validation_error(
            "_bound_raw_counts", "residue_counts must contain exactly modulus rows"
        )

    maximum_count_digits = _maximum_count_digits(item_count)
    for count in counts:
        if not isinstance(count, str):
            continue
        if len(count) > maximum_count_digits:
            raise _validation_error(
                "_bound_raw_counts",
                "residue count exceeds the source-derived multiplicity bound",
            )
        result_bytes += len(count) + 3
        if result_bytes > MAX_RESIDUE_PROFILE_RESULT_BYTES:
            raise _validation_error(
                "_bound_raw_counts",
                "subset-sum residue profile exceeds the 4 MiB result bound",
            )
    return result_bytes


def _bound_raw_witnesses(
    witnesses: object,
    *,
    expected_rows: int | None,
    item_count: int,
    result_bytes: int,
) -> None:
    if not isinstance(witnesses, (list, tuple)):
        return
    if len(witnesses) > MAX_RESIDUE_PROFILE_MODULUS:
        raise _validation_error(
            "_bound_raw_witnesses",
            "residue_witnesses exceeds the bounded result cardinality",
        )
    if expected_rows is not None and len(witnesses) != expected_rows:
        raise _validation_error(
            "_bound_raw_witnesses",
            "residue_witnesses must contain exactly modulus rows",
        )

    index_slots = 0
    index_digits = len(str(max(item_count - 1, 0)))
    result_bytes += len(witnesses) * 24
    for witness in witnesses:
        if isinstance(witness, IndexSubset):
            indices: list[object] | tuple[object, ...] = witness.indices
        elif isinstance(witness, Mapping):
            raw_indices = witness.get("indices")
            if not isinstance(raw_indices, (list, tuple)):
                continue
            indices = raw_indices
        else:
            continue
        if len(indices) > item_count:
            raise _validation_error(
                "_bound_raw_witnesses",
                "one residue witness exceeds the retained source length",
            )
        for index in indices:
            if type(index) is int and not 0 <= index < item_count:
                raise _validation_error(
                    "_bound_raw_witnesses",
                    "residue witness index lies outside the retained source",
                )
        index_slots += len(indices)
        if index_slots > MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS:
            raise _validation_error(
                "_bound_raw_witnesses",
                "residue witnesses exceed the 250,000 index-slot storage bound",
            )
    result_bytes += index_slots * (index_digits + 1)
    if result_bytes > MAX_RESIDUE_PROFILE_RESULT_BYTES:
        raise _validation_error(
            "_bound_raw_witnesses",
            "subset-sum residue profile exceeds the 4 MiB result bound",
        )


class SubsetSumResidueProfileRequest(StrictModel):
    """Request a complete indexed-subset multiplicity profile in ``Z/mZ``."""

    source: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_RESIDUE_PROFILE_ITEMS)),
    ] = Field(
        description=(
            "A materialized indexed integer tuple. At most 4,095 positions are "
            "admitted; every integer carries at most 32,768 digits, and the "
            "derived DP, witness, and result budgets are checked jointly with "
            "the modulus."
        )
    )
    modulus: int = Field(
        ge=1,
        le=MAX_RESIDUE_PROFILE_MODULUS,
        strict=True,
        description=(
            "Positive modulus. The complete result has exactly this many "
            "residue counts, in order from residue 0 through modulus-1."
        ),
    )
    include_empty_subset: StrictBool = Field(
        description=(
            "Whether the empty index subset contributes one representation "
            "to residue 0."
        )
    )
    include_witnesses: StrictBool = Field(
        default=False,
        description=(
            "Return one canonical attaining index subset for every reachable "
            "residue, minimizing sum(2**i for i in I); witness-bearing requests "
            "use a stricter output budget."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_source(cls, value: object) -> object:
        """Reject impossible source containers before nested scalar parsing.

        Running a before validator moves field validation into Python
        mode, where decoded JSON arrays no longer coerce to the declared
        tuple shapes; normalize the source value list to a tuple on a
        copied path so JSON invocation keeps working while the stored
        sequence stays canonical.
        """

        value = canonicalize_json_containers(value)

        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_source = prepared.get("source")
        if isinstance(raw_source, Mapping):
            source = dict(raw_source)
            values = source.get("items")
            if isinstance(values, list):
                source["items"] = tuple(values)
            prepared["source"] = source
        _raw_source_shape(prepared.get("source"))
        return prepared

    @model_validator(mode="after")
    def require_bounded_complete_profile(self) -> Self:
        item_count = len(self.source.items)
        if item_count + 1 > MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS:
            raise _validation_error(
                "require_bounded_complete_profile",
                "subset multiplicities exceed the 4,096-bit intermediate bound",
            )

        oversized = next(
            (
                index
                for index, value in enumerate(self.source.items)
                if len(value.lstrip("-")) > MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS
            ),
            None,
        )
        if oversized is not None:
            raise _validation_error(
                "require_bounded_complete_profile",
                f"source value at index {oversized} exceeds the 32,768-digit "
                "input bound",
            )

        dp_cells = item_count * self.modulus
        if dp_cells > MAX_RESIDUE_PROFILE_DP_CELLS:
            raise _validation_error(
                "require_bounded_complete_profile",
                "subset-sum residue DP exceeds the 1,000,000-cell work bound",
            )

        if (
            self.include_witnesses
            and dp_cells > MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS
        ):
            raise _validation_error(
                "require_bounded_complete_profile",
                "residue witnesses exceed the 250,000 index-slot storage bound",
            )

        result_bytes = _estimated_result_bytes(
            self.source,
            self.modulus,
            include_witnesses=self.include_witnesses,
        )
        if result_bytes > MAX_RESIDUE_PROFILE_RESULT_BYTES:
            raise _validation_error(
                "require_bounded_complete_profile",
                "subset-sum residue profile exceeds the 4 MiB result bound",
            )
        return self


class SubsetSumResidueProfileResult(StrictModel):
    """A complete exact residue profile bound to its indexed source."""

    source: Annotated[
        IndexedIntegerSequence,
        WithJsonSchema(indexed_sequence_item_ceiling(MAX_RESIDUE_PROFILE_ITEMS)),
    ]
    modulus: int = Field(ge=1, le=MAX_RESIDUE_PROFILE_MODULUS, strict=True)
    include_empty_subset: StrictBool
    include_witnesses: StrictBool
    residue_counts: tuple[ResidueMultiplicity, ...] = Field(
        min_length=1,
        max_length=MAX_RESIDUE_PROFILE_MODULUS,
        description=(
            "Exact subset multiplicities indexed by residues 0 through modulus-1."
        ),
    )
    residue_witnesses: tuple[IndexSubset | None, ...] | None = Field(
        default=None,
        description=(
            "When requested, one attaining index subset minimizing "
            "sum(2**i for i in I) per reachable residue, and null for each "
            "unreachable residue."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        """Reject oversized forged result containers before nested parsing."""

        value = canonicalize_json_containers(value)

        if not isinstance(value, Mapping):
            return value

        prepared: dict[str, object] = dict(value)
        raw_source = prepared.get("source")
        if isinstance(raw_source, Mapping):
            source = dict(raw_source)
            source_values = source.get("items")
            if isinstance(source_values, list):
                source["items"] = tuple(source_values)
            prepared["source"] = source
        raw_counts = prepared.get("residue_counts")
        if isinstance(raw_counts, list):
            prepared["residue_counts"] = tuple(raw_counts)
        raw_witnesses = prepared.get("residue_witnesses")
        if isinstance(raw_witnesses, list):
            witnesses: list[object] = []
            for raw_witness in raw_witnesses:
                if isinstance(raw_witness, Mapping):
                    witness = dict(raw_witness)
                    raw_indices = witness.get("indices")
                    if isinstance(raw_indices, list):
                        witness["indices"] = tuple(raw_indices)
                    witnesses.append(witness)
                else:
                    witnesses.append(raw_witness)
            prepared["residue_witnesses"] = tuple(witnesses)

        source_shape = _raw_source_shape(prepared.get("source"))
        item_count = (
            source_shape[0]
            if source_shape is not None
            else MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS - 1
        )
        result_bytes = source_shape[1] if source_shape is not None else 1_024
        raw_modulus = prepared.get("modulus")
        expected_rows = (
            raw_modulus
            if type(raw_modulus) is int
            and 1 <= raw_modulus <= MAX_RESIDUE_PROFILE_MODULUS
            else None
        )

        result_bytes = _bound_raw_counts(
            prepared.get("residue_counts"),
            expected_rows=expected_rows,
            item_count=item_count,
            result_bytes=result_bytes,
        )
        _bound_raw_witnesses(
            prepared.get("residue_witnesses"),
            expected_rows=expected_rows,
            item_count=item_count,
            result_bytes=result_bytes,
        )
        return prepared

    @model_validator(mode="after")
    def require_structural_shape(self) -> Self:
        if self.include_witnesses != (self.residue_witnesses is not None):
            raise _validation_error(
                "result_shape",
                "residue_witnesses presence must match include_witnesses",
            )
        if len(self.residue_counts) != self.modulus:
            raise _validation_error(
                "result_shape",
                "residue_counts must contain exactly modulus rows",
            )
        if self.residue_witnesses is not None:
            if len(self.residue_witnesses) != self.modulus:
                raise _validation_error(
                    "result_shape",
                    "residue_witnesses must contain exactly modulus rows",
                )
            item_count = len(self.source.items)
            for witness in self.residue_witnesses:
                if witness is not None and any(
                    index >= item_count for index in witness.indices
                ):
                    raise _validation_error(
                        "result_shape",
                        "residue witness index lies outside the retained source",
                    )
        return self

    @classmethod
    def _from_kernel(
        cls,
        request: SubsetSumResidueProfileRequest,
        residue_counts: tuple[ResidueMultiplicity, ...],
        residue_witnesses: tuple[IndexSubset | None, ...] | None,
    ) -> Self:
        """Construct output whose mathematical invariants the kernel established."""

        return cls.model_construct(
            source=request.source,
            modulus=request.modulus,
            include_empty_subset=request.include_empty_subset,
            include_witnesses=request.include_witnesses,
            residue_counts=residue_counts,
            residue_witnesses=residue_witnesses,
        )


def _indices_from_mask(mask: int, item_count: int) -> tuple[int, ...]:
    return tuple(index for index in range(item_count) if mask & (1 << index))


def _compute_residue_profile(
    request: SubsetSumResidueProfileRequest,
) -> tuple[
    tuple[ResidueMultiplicity, ...],
    tuple[IndexSubset | None, ...] | None,
]:
    """Run the admitted dense recurrence over residues.

    Witnesses are canonical by minimizing ``sum(2**i for i in I)``.  This is a
    normalization only, not a minimum-cardinality claim.
    """
    modulus = request.modulus
    counts = [0] * modulus
    witness_masks: list[int | None] | None = (
        [None] * modulus if request.include_witnesses else None
    )
    if request.include_empty_subset:
        counts[0] = 1
        if witness_masks is not None:
            witness_masks[0] = 0

    for index, raw_value in enumerate(request.source.items):
        residue = parse_canonical_integer(raw_value) % modulus
        next_counts = counts.copy()
        next_witness_masks = witness_masks.copy() if witness_masks is not None else None
        bit = 1 << index

        if not request.include_empty_subset:
            next_counts[residue] += 1
            if next_witness_masks is not None:
                current = next_witness_masks[residue]
                if current is None or bit < current:
                    next_witness_masks[residue] = bit

        for prior_residue, multiplicity in enumerate(counts):
            if multiplicity == 0:
                continue
            target = (prior_residue + residue) % modulus
            next_counts[target] += multiplicity
            if next_witness_masks is not None and witness_masks is not None:
                prior_mask = witness_masks[prior_residue]
                if prior_mask is None:
                    raise AssertionError("positive multiplicity must have a witness")
                candidate = prior_mask | bit
                current = next_witness_masks[target]
                if current is None or candidate < current:
                    next_witness_masks[target] = candidate

        counts = next_counts
        witness_masks = next_witness_masks

    residue_counts = tuple(format_canonical_integer(count) for count in counts)
    if witness_masks is None:
        return residue_counts, None
    residue_witnesses = tuple(
        None
        if mask is None
        else IndexSubset(indices=_indices_from_mask(mask, len(request.source.items)))
        for mask in witness_masks
    )
    return residue_counts, residue_witnesses


def compute_subset_sum_residue_profile(
    request: SubsetSumResidueProfileRequest,
) -> SubsetSumResidueProfileResult:
    """Return every exact indexed-subset multiplicity modulo ``m``."""
    residue_counts, residue_witnesses = _compute_residue_profile(request)
    return SubsetSumResidueProfileResult._from_kernel(
        request,
        residue_counts=residue_counts,
        residue_witnesses=residue_witnesses,
    )


def _verify_subset_sum_residue_profile(result: SubsetSumResidueProfileResult) -> bool:
    """Verify an independently supplied residue profile within its envelope."""

    try:
        request = SubsetSumResidueProfileRequest(
            source=result.source,
            modulus=result.modulus,
            include_empty_subset=result.include_empty_subset,
            include_witnesses=result.include_witnesses,
        )
        expected_counts, expected_witnesses = _compute_residue_profile(request)
    except ValueError:
        return False
    return (
        result.residue_counts == expected_counts
        and result.residue_witnesses == expected_witnesses
    )


__all__ = [
    "MAX_RESIDUE_PROFILE_DP_CELLS",
    "MAX_RESIDUE_PROFILE_INPUT_INTEGER_DIGITS",
    "MAX_RESIDUE_PROFILE_ITEMS",
    "MAX_RESIDUE_PROFILE_MODULUS",
    "MAX_RESIDUE_PROFILE_MULTIPLICITY_BITS",
    "MAX_RESIDUE_PROFILE_RESULT_BYTES",
    "MAX_RESIDUE_PROFILE_WITNESS_INDEX_SLOTS",
    "SubsetSumResidueProfileRequest",
    "SubsetSumResidueProfileResult",
    "compute_subset_sum_residue_profile",
]


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)
