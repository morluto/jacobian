"""Typed exact target subset-sum operation."""

from __future__ import annotations

from collections.abc import Mapping
from math import gcd
from typing import Literal, Self

from pydantic import Field, StrictBool, model_validator

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.math.additive_combinatorics._subset_sum_target_kernel import (
    _solve_subset_sum_target,
)
from jacobian.math.additive_combinatorics.values import (
    IndexedIntegerSequence,
    IndexSubset,
)

MAX_SUBSET_SUM_INTEGER_DIGITS = 256
MAX_SUBSET_SUM_REACHABLE_STATES = 65_536
MAX_SUBSET_SUM_TRANSITIONS_PER_PASS = 500_000
MAX_SUBSET_SUM_TOTAL_TRANSITIONS = 2 * MAX_SUBSET_SUM_TRANSITIONS_PER_PASS
MAX_SUBSET_SUM_SOURCE_WIRE_BYTES = 4 * 1024 * 1024
MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS = 262
_SUBSET_COUNT_EXACT_ITEM_LIMIT = MAX_SUBSET_SUM_REACHABLE_STATES.bit_length() - 1


def _require_integer_digits(value: str, label: str, maximum_digits: int) -> None:
    if len(value.lstrip("-")) > maximum_digits:
        raise ValueError(f"{label} exceeds the {maximum_digits}-digit bound")


def _require_target_within_subset_sum_width(
    target: str,
    values: tuple[int, ...],
) -> None:
    """Bound the target by the widest subset sum the parsed source can attain.

    Every subset sum lies in ``[negative_span, positive_span]``, so its
    canonical width never exceeds the wider endpoint of that attained range.
    """

    positive_span = sum(value for value in values if value > 0)
    negative_span = sum(value for value in values if value < 0)
    attainable_width = max(
        len(format_canonical_integer(positive_span).lstrip("-")),
        len(format_canonical_integer(negative_span).lstrip("-")),
    )
    if len(target.lstrip("-")) > attainable_width:
        raise ValueError(
            f"target exceeds the {attainable_width}-digit attainable "
            "subset-sum width bound"
        )


def _raw_source_item_count(source: object) -> int | None:
    """Bound one raw source before Pydantic parses its canonical integers."""

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
    if item_count > MAX_SUBSET_SUM_TRANSITIONS_PER_PASS:
        raise ValueError(
            "subset-sum request exceeds the "
            f"{MAX_SUBSET_SUM_TOTAL_TRANSITIONS:,}-transition complete-call "
            "bound across computation and source-binding replay"
        )

    source_wire_bound = 64
    for value in values:
        if not isinstance(value, str):
            continue
        digit_count = len(value) - value.startswith("-")
        if digit_count > MAX_SUBSET_SUM_INTEGER_DIGITS:
            _require_integer_digits(value, "source item", MAX_SUBSET_SUM_INTEGER_DIGITS)
        source_wire_bound += len(value) + 4
        if source_wire_bound > MAX_SUBSET_SUM_SOURCE_WIRE_BYTES:
            raise ValueError("subset-sum source exceeds the 4 MiB wire-size bound")
    return item_count


def _require_admitted_work(values: tuple[int, ...], allow_empty: bool) -> None:
    """Bound every reachable-state update before the exhaustive dynamic program."""

    prefix_span = 0
    prefix_gcd = 0
    previous_state_bound = 1 if allow_empty else 0
    transition_bound = 0

    for item_count, value in enumerate(values, start=1):
        transition_bound += previous_state_bound + (0 if allow_empty else 1)
        if transition_bound > MAX_SUBSET_SUM_TRANSITIONS_PER_PASS:
            raise ValueError(
                "subset-sum request exceeds the "
                f"{MAX_SUBSET_SUM_TOTAL_TRANSITIONS:,}-transition complete-call "
                "bound across computation and source-binding replay"
            )

        prefix_span += abs(value)
        prefix_gcd = gcd(prefix_gcd, abs(value))
        lattice_point_bound = 1 if prefix_gcd == 0 else prefix_span // prefix_gcd + 1
        subset_count_bound = (
            (1 << item_count) - (0 if allow_empty else 1)
            if item_count <= _SUBSET_COUNT_EXACT_ITEM_LIMIT
            else MAX_SUBSET_SUM_REACHABLE_STATES + 1
        )
        previous_state_bound = min(subset_count_bound, lattice_point_bound)
        if previous_state_bound > MAX_SUBSET_SUM_REACHABLE_STATES:
            raise ValueError(
                "subset-sum request exceeds the 65,536-reachable-state bound"
            )


class SubsetSumTargetRequest(StrictModel):
    """One bounded indexed integer sequence and exact target."""

    source: IndexedIntegerSequence = Field(
        description=(
            "The materialized indexed integers available for selection; every item "
            "has at most 256 digits and the retained source has a 4 MiB wire bound."
        )
    )
    target: CanonicalInteger = Field(
        description=(
            "The canonical signed decimal sum to decide exactly; its digit "
            "width must fit the attainable subset-sum width derived from this "
            "source, which stays at most 262 digits for every admitted source."
        )
    )
    allow_empty_subset: StrictBool = Field(
        description="Whether the empty index subset is an admissible witness."
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_request(cls, value: object) -> object:
        """Reject oversized raw strings and containers before nested parsing.

        Running a before validator moves field validation into Python
        mode, where decoded JSON arrays no longer coerce to the declared
        tuple shapes; normalize the source value list to a tuple on a
        copied path so JSON invocation keeps working while the stored
        sequence stays canonical.
        """

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
        _raw_source_item_count(prepared.get("source"))
        raw_target = prepared.get("target")
        if isinstance(raw_target, str):
            # Admitted sources hold at most 500,000 items of 256 digits each,
            # so no attainable subset sum is wider than this raw bound.
            _require_integer_digits(
                raw_target, "target", MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS
            )
        return prepared

    @model_validator(mode="after")
    def require_bounded_exact_search(self) -> Self:
        for value in self.source.items:
            _require_integer_digits(value, "source item", MAX_SUBSET_SUM_INTEGER_DIGITS)

        # Exact ASCII digits plus conservative JSON string/container overhead.
        # Retaining this source and a maximal 65,536-index witness stays below
        # Jacobian's 10 MiB canonical result limit.
        source_wire_bound = 64 + sum(len(value) + 4 for value in self.source.items)
        if source_wire_bound > MAX_SUBSET_SUM_SOURCE_WIRE_BYTES:
            raise ValueError("subset-sum source exceeds the 4 MiB wire-size bound")

        values = tuple(parse_canonical_integer(value) for value in self.source.items)
        target = parse_canonical_integer(self.target)
        _require_target_within_subset_sum_width(self.target, values)
        if not (self.allow_empty_subset and target == 0):
            # The empty witness already resolves this target exactly, so the
            # exhaustive reachable-state bound applies only to real searches.
            _require_admitted_work(values, self.allow_empty_subset)
        return self


class SubsetSumTargetResult(StrictModel):
    """An exact source-bound target decision with its canonical witness."""

    source: IndexedIntegerSequence
    target: CanonicalInteger
    allow_empty_subset: StrictBool
    status: Literal["ATTAINED", "NOT_ATTAINED"] = Field(
        description="The complete exact decision inside the admitted search envelope."
    )
    witness: IndexSubset | None = Field(
        default=None,
        description=(
            "The canonical attaining index subset exactly when status is ATTAINED."
        ),
    )
    reconstructed_sum: CanonicalInteger | None = Field(
        default=None,
        description="The exact sum reconstructed from an attaining witness.",
    )

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        """Reject oversized forged result values before nested parsing."""

        if not isinstance(value, Mapping):
            return value
        prepared: dict[str, object] = dict(value)
        raw_source = prepared.get("source")
        if isinstance(raw_source, Mapping):
            source = dict(raw_source)
            raw_values = source.get("items")
            if isinstance(raw_values, list):
                source["items"] = tuple(raw_values)
            prepared["source"] = source
        raw_witness = prepared.get("witness")
        if isinstance(raw_witness, Mapping):
            witness = dict(raw_witness)
            raw_indices = witness.get("indices")
            if isinstance(raw_indices, list):
                witness["indices"] = tuple(raw_indices)
            prepared["witness"] = witness
        item_count = _raw_source_item_count(prepared.get("source"))
        raw_target = prepared.get("target")
        if isinstance(raw_target, str):
            _require_integer_digits(
                raw_target, "target", MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS
            )
        raw_sum = prepared.get("reconstructed_sum")
        if isinstance(raw_sum, str) and (
            len(raw_sum) - raw_sum.startswith("-") > MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS
        ):
            raise ValueError("reconstructed_sum exceeds the 262-digit result bound")

        raw_witness = prepared.get("witness")
        if isinstance(raw_witness, IndexSubset):
            indices: list[object] | tuple[object, ...] = raw_witness.indices
        elif isinstance(raw_witness, Mapping):
            raw_indices = raw_witness.get("indices")
            if not isinstance(raw_indices, (list, tuple)):
                return prepared
            indices = raw_indices
        else:
            return prepared
        if len(indices) > MAX_SUBSET_SUM_REACHABLE_STATES:
            raise ValueError("subset-sum witness exceeds the 65,536-index result bound")
        maximum_index = (
            item_count
            if item_count is not None
            else MAX_SUBSET_SUM_TRANSITIONS_PER_PASS
        )
        for index in indices:
            if type(index) is int and not 0 <= index < maximum_index:
                raise ValueError("subset-sum witness index lies outside its source")
        return prepared

    @model_validator(mode="after")
    def bind_decision_to_source(self) -> Self:
        request = SubsetSumTargetRequest(
            source=self.source,
            target=self.target,
            allow_empty_subset=self.allow_empty_subset,
        )
        values = tuple(parse_canonical_integer(value) for value in request.source.items)
        target = parse_canonical_integer(request.target)
        expected_indices = _solve_subset_sum_target(
            values,
            target,
            allow_empty_subset=request.allow_empty_subset,
        )

        if expected_indices is None:
            if self.status != "NOT_ATTAINED":
                raise ValueError("status must report the exhaustive target decision")
            if self.witness is not None or self.reconstructed_sum is not None:
                raise ValueError("an unattained target cannot carry a witness or sum")
            return self

        if self.status != "ATTAINED":
            raise ValueError("status must report the exhaustive target decision")
        expected_witness = IndexSubset(indices=expected_indices)
        if self.witness != expected_witness:
            raise ValueError("witness must be the canonical attaining index subset")
        expected_sum = sum(values[index] for index in expected_indices)
        expected_sum_wire = format_canonical_integer(expected_sum)
        if self.reconstructed_sum != expected_sum_wire:
            raise ValueError("reconstructed_sum must equal the witness sum")
        if expected_sum != target:
            raise ValueError("witness sum must equal the requested target")
        return self


def solve_subset_sum_target_request(
    request: SubsetSumTargetRequest,
) -> SubsetSumTargetResult:
    """Decide one admitted exact target and return its canonical witness."""

    values = tuple(parse_canonical_integer(value) for value in request.source.items)
    target = parse_canonical_integer(request.target)
    indices = _solve_subset_sum_target(
        values,
        target,
        allow_empty_subset=request.allow_empty_subset,
    )
    if indices is None:
        return SubsetSumTargetResult(
            source=request.source,
            target=request.target,
            allow_empty_subset=request.allow_empty_subset,
            status="NOT_ATTAINED",
        )

    reconstructed_sum = sum(values[index] for index in indices)
    return SubsetSumTargetResult(
        source=request.source,
        target=request.target,
        allow_empty_subset=request.allow_empty_subset,
        status="ATTAINED",
        witness=IndexSubset(indices=indices),
        reconstructed_sum=format_canonical_integer(reconstructed_sum),
    )
