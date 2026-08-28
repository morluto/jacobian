"""Typed exact target subset-sum operation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self

from pydantic import Field, StrictBool, StringConstraints, model_validator
from pydantic.json_schema import WithJsonSchema
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel, canonicalize_json_containers
from jacobian.canonical import format_canonical_integer, parse_canonical_integer
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.combinatorics.additive._subset_sum_target_kernel import (
    _attained_sum_interval,
    _solve_subset_sum_target,
)
from jacobian.math.combinatorics.additive.values import (
    IndexedIntegerSequence,
    IndexSubset,
    indexed_sequence_item_ceiling,
)

MAX_SUBSET_SUM_INTEGER_DIGITS = 256
MAX_SUBSET_SUM_REACHABLE_STATES = 65_536
MAX_SUBSET_SUM_TRANSITIONS_PER_PASS = 500_000
# One complete public call performs two charged DP-equivalent passes: request
# admission and the solver kernel. Independent claims are checked explicitly
# by the owner verifier, rather than as a result-model side effect.
MAX_SUBSET_SUM_COMPLETE_CALL_PASSES = 2
MAX_SUBSET_SUM_TOTAL_TRANSITIONS = (
    MAX_SUBSET_SUM_COMPLETE_CALL_PASSES * MAX_SUBSET_SUM_TRANSITIONS_PER_PASS
)
MAX_SUBSET_SUM_SOURCE_WIRE_BYTES = 4 * 1024 * 1024
MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS = 262

_SubsetSumTargetScalar = Annotated[
    str,
    StringConstraints(
        pattern=rf"^(?:0|-?[1-9][0-9]{{0,{MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS - 1}}})$",
        strict=True,
        max_length=MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS + 1,
    ),
]
"""Canonical signed integer whose magnitude carries at most 262 digits.

The absolute digit ceiling is published as a JSON Schema ``pattern``, so
schema-driven clients see the enforced domain; only a negative 262-digit
value needs the extra sign character of ``maxLength``.
"""

_SubsetSumTargetSource = Annotated[
    IndexedIntegerSequence,
    WithJsonSchema(
        indexed_sequence_item_ceiling(
            MAX_SUBSET_SUM_TRANSITIONS_PER_PASS,
            maximum_item_digits=MAX_SUBSET_SUM_INTEGER_DIGITS,
        )
    ),
]
"""The shared sequence narrowed to this operation's admitted source items.

Request and result admission both reject any item wider than
``MAX_SUBSET_SUM_INTEGER_DIGITS`` digits, so the published schema encodes
that absolute per-item ceiling instead of the shared 32,768-digit envelope;
validation itself stays with the canonical sequence value.
"""


def _require_integer_digits(value: str, label: str, maximum_digits: int) -> None:
    if len(value.lstrip("-")) > maximum_digits:
        raise _validation_error(
            "_require_integer_digits",
            f"{label} exceeds the {maximum_digits}-digit bound",
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
        raise _validation_error(
            "_raw_source_item_count",
            "subset-sum request exceeds the "
            f"{MAX_SUBSET_SUM_TOTAL_TRANSITIONS:,}-transition complete-call "
            "bound across admission and computation",
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
            raise _validation_error(
                "_raw_source_item_count",
                "subset-sum source exceeds the 4 MiB wire-size bound",
            )
    return item_count


def _require_admitted_work(
    values: tuple[int, ...],
    target: int,
    *,
    allow_empty_subset: bool,
) -> None:
    """Bound the kernel's insert-only passes with their exact reachable sums.

    Every admissible subset sum lies in the attained interval from
    ``_attained_sum_interval``, so a target outside that closed interval is
    exactly unattainable before any state exists; with the empty subset
    inadmissible the interval excludes its zero, so strictly one-signed
    sources resolve a zero target without expansion. Admission resolves such
    targets with no expansion, and the kernel returns the same decision
    without building any state.

    Otherwise admission simulates the same reachable-sum growth the kernel
    performs and stops at the first source prefix whose canonical witness
    resolves the target. A resolved request is bounded only by the work
    through that prefix, because every witness inside a resolving prefix
    carries a strictly smaller incidence mask than any witness of the later
    expansion; the resolving scan itself is charged before its prefix is
    accepted, so the charged work includes every scanned state. A request the
    prefix simulation never resolves must fit the exhaustive reachable-state and
    transition bounds across the whole source before execution.

    The public path performs this scan once for request admission and runs the
    kernel once for computation. Each pass inspects the same reachable states
    through the same prefix, so charging two identical per-pass ceilings bounds
    every accepted complete call by
    ``MAX_SUBSET_SUM_TOTAL_TRANSITIONS``.
    """

    lower, upper = _attained_sum_interval(
        values,
        allow_empty_subset=allow_empty_subset,
    )
    if not lower <= target <= upper:
        # Outside the attained interval no admissible subset sum can equal
        # the target.
        return
    states: set[int] = {0} if allow_empty_subset else set()
    if target in states:
        # The retained empty witness is already the globally smallest mask.
        return
    transitions = 0
    for value in values:
        if not allow_empty_subset and value == target:
            return
        transitions += len(states) + (0 if allow_empty_subset else 1)
        if transitions > MAX_SUBSET_SUM_TRANSITIONS_PER_PASS:
            raise OperationDomainValidationError(
                location=("source",),
                code="additive_combinatorics.subset_sum.transition_bound",
                message=(
                    "subset-sum request exceeds the "
                    f"{MAX_SUBSET_SUM_TOTAL_TRANSITIONS:,}-transition complete-call "
                    "bound across admission and computation"
                ),
            )
        if any(subtotal + value == target for subtotal in states):
            return

        grown = {subtotal + value for subtotal in states}
        if not allow_empty_subset and value not in states:
            grown.add(value)
        grown.update(states)
        if len(grown) > MAX_SUBSET_SUM_REACHABLE_STATES:
            raise OperationDomainValidationError(
                location=("source",),
                code="additive_combinatorics.subset_sum.state_bound",
                message="subset-sum request exceeds the 65,536-reachable-state bound",
            )
        states = grown


class SubsetSumTargetRequest(StrictModel):
    """One bounded indexed integer sequence and exact target."""

    source: _SubsetSumTargetSource = Field(
        description=(
            "The materialized indexed integers available for selection; "
            "request admission bounds this operation to at most "
            f"{MAX_SUBSET_SUM_TRANSITIONS_PER_PASS:,} items of at most "
            f"{MAX_SUBSET_SUM_INTEGER_DIGITS} digits each with a 4 MiB wire "
            "bound across the whole source."
        )
    )
    target: _SubsetSumTargetScalar = Field(
        description=(
            "The canonical signed decimal sum to decide exactly; its "
            f"magnitude carries at most {MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS} "
            "digits, and a value outside this source's attained subset-sum "
            "interval is decided exactly as unattained without expansion."
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
        _raw_source_item_count(prepared.get("source"))
        raw_target = prepared.get("target")
        if isinstance(raw_target, str):
            # Admitted sources hold at most 500,000 items of 256 digits each,
            # so no attainable subset sum is wider than this raw bound.
            _require_integer_digits(
                raw_target, "target", MAX_SUBSET_SUM_RECONSTRUCTED_DIGITS
            )
        return prepared


class SubsetSumTargetResult(StrictModel):
    """An exact source-bound target decision with its canonical witness."""

    source: _SubsetSumTargetSource
    target: _SubsetSumTargetScalar
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

    @model_validator(mode="after")
    def require_result_branch_shape(self) -> Self:
        if self.status == "ATTAINED":
            if self.witness is None or self.reconstructed_sum is None:
                raise _validation_error(
                    "result_branch_shape",
                    "an ATTAINED result requires a witness and reconstructed sum",
                )
        elif self.witness is not None or self.reconstructed_sum is not None:
            raise _validation_error(
                "result_branch_shape",
                "a NOT_ATTAINED result cannot carry a witness or reconstructed sum",
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def bound_raw_result(cls, value: object) -> object:
        """Reject oversized forged result values before nested parsing."""

        value = canonicalize_json_containers(value)

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
            raise _validation_error(
                "bound_raw_result",
                "reconstructed_sum exceeds the 262-digit result bound",
            )

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
            raise _validation_error(
                "bound_raw_result",
                "subset-sum witness exceeds the 65,536-index result bound",
            )
        maximum_index = (
            item_count
            if item_count is not None
            else MAX_SUBSET_SUM_TRANSITIONS_PER_PASS
        )
        for index in indices:
            if type(index) is int and not 0 <= index < maximum_index:
                raise _validation_error(
                    "bound_raw_result",
                    "subset-sum witness index lies outside its source",
                )
        return prepared

    @classmethod
    def _from_kernel(
        cls,
        request: SubsetSumTargetRequest,
        indices: tuple[int, ...] | None,
    ) -> Self:
        """Construct trusted output without replaying the target search."""

        if indices is None:
            return cls.model_construct(
                source=request.source,
                target=request.target,
                allow_empty_subset=request.allow_empty_subset,
                status="NOT_ATTAINED",
                witness=None,
                reconstructed_sum=None,
            )
        values = tuple(parse_canonical_integer(value) for value in request.source.items)
        return cls.model_construct(
            source=request.source,
            target=request.target,
            allow_empty_subset=request.allow_empty_subset,
            status="ATTAINED",
            witness=IndexSubset(indices=indices),
            reconstructed_sum=format_canonical_integer(
                sum(values[index] for index in indices)
            ),
        )


def solve_subset_sum_target_request(
    request: SubsetSumTargetRequest,
) -> SubsetSumTargetResult:
    """Decide one admitted exact target and return its canonical witness."""

    _admit_subset_sum_target(request)
    values = tuple(parse_canonical_integer(value) for value in request.source.items)
    target = parse_canonical_integer(request.target)
    indices = _solve_subset_sum_target(
        values,
        target,
        allow_empty_subset=request.allow_empty_subset,
    )
    if indices is None:
        return SubsetSumTargetResult._from_kernel(request, None)

    return SubsetSumTargetResult._from_kernel(request, indices)


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"additive_combinatorics.{reason}", message)


def _admit_subset_sum_target(request: SubsetSumTargetRequest) -> None:
    """Admit one native target query before invoking the subset-sum kernel."""

    source_wire_bound = 64 + sum(len(value) + 4 for value in request.source.items)
    if source_wire_bound > MAX_SUBSET_SUM_SOURCE_WIRE_BYTES:
        raise OperationDomainValidationError(
            location=("source",),
            code="additive_combinatorics.subset_sum.source_wire_bound",
            message="subset-sum source exceeds the 4 MiB wire-size bound",
        )
    for value in request.source.items:
        if len(value.lstrip("-")) > MAX_SUBSET_SUM_INTEGER_DIGITS:
            raise OperationDomainValidationError(
                location=("source",),
                code="additive_combinatorics.subset_sum.integer_bound",
                message=(
                    "source item exceeds the "
                    f"{MAX_SUBSET_SUM_INTEGER_DIGITS}-digit bound"
                ),
            )
    values = tuple(parse_canonical_integer(value) for value in request.source.items)
    target = parse_canonical_integer(request.target)
    _require_admitted_work(
        values,
        target,
        allow_empty_subset=request.allow_empty_subset,
    )
