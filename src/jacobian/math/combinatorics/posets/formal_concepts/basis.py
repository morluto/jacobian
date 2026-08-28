"""Canonical values and bounds for exact finite implication bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, StrictInt, model_validator

from jacobian._models import StrictModel
from jacobian.canonical import (
    CanonicalizationError,
    CanonicalLimits,
    encode_strict_json,
)
from jacobian.math.combinatorics.posets.formal_concepts.values import (
    MAX_IMPLICATION_MEMBERSHIPS,
    MAX_IMPLICATIONS,
    MAX_OBJECTS,
    FiniteAttributeImplicationSystem,
    FormalContext,
)

# Admission bounds three operation-specific quantities instead of a coarse
# attribute count (AGENTS.md, "Mathematical boundedness is a proof
# obligation"):
#
#   * candidate states N = 2^attribute_count -- the exhaustive closure-matrix
#     carrier and the finite candidate space of the lectic scan.  The fixed
#     MAX_DG_CANDIDATE_STATES is a documented conservative fallback that keeps
#     the exact admission probe itself (one context closure per candidate
#     state) bounded before execution;
#   * logical work -- the plan reports the exact closure-matrix and lectic
#     enumeration counts. The kernel consumes that carrier, then performs one
#     closure-equivalence pass over the constructed basis. The carrier-fit
#     check below keeps that basis inside MAX_IMPLICATIONS rows and
#     MAX_IMPLICATION_MEMBERSHIPS memberships, so
#     every admitted query also stays under MAX_CANONICAL_CLOSURE_WORK;
#   * serialized result bytes -- a worst-case payload shaped by the probed
#     basis size with full-width rows, measured through the strict-JSON
#     canonical encoder.
#
# Contexts whose exact basis exceeds the canonical implication carrier
# are rejected on the probe, so every accepted request returns the declared
# typed result.
MAX_DG_CANDIDATE_STATES = 4_096
MAX_DG_ATTRIBUTES = MAX_DG_CANDIDATE_STATES.bit_length() - 1
MAX_DG_LOGICAL_WORK = 1 << 30
MAX_DG_RESULT_BYTES = 1 * 1_024 * 1_024

_DGAttributeIndex = Annotated[
    StrictInt,
    Field(ge=0, le=MAX_DG_ATTRIBUTES - 1),
]
_DGCandidateState = Annotated[
    StrictInt,
    Field(ge=0, lt=MAX_DG_CANDIDATE_STATES),
]


def _subset_for_state(state: int, attribute_count: int) -> tuple[int, ...]:
    return tuple(
        attribute for attribute in range(attribute_count) if state & (1 << attribute)
    )


def _basis_attribute_labels(attribute_count: int) -> tuple[str, ...]:
    """Return canonical labels for source-index coordinates."""

    return tuple(f"m{attribute}" for attribute in range(attribute_count))


def _require_dg_subset(
    name: str,
    subset: tuple[int, ...],
    attribute_count: int,
) -> None:
    if subset != tuple(sorted(set(subset))):
        raise ValueError(f"{name} must be sorted and duplicate-free")
    if any(attribute >= attribute_count for attribute in subset):
        raise ValueError(f"{name} contains an attribute outside the source context")


def _context_closure_masks(context: FormalContext) -> tuple[tuple[int, ...], int]:
    """Return every candidate state's context closure plus row intersections."""

    object_rows = [0] * len(context.objects)
    for object_index, attribute_index in context.incidence:
        object_rows[object_index] |= 1 << attribute_index
    full_mask = (1 << len(context.attributes)) - 1
    masks: list[int] = []
    row_intersections = 0
    for state in range(1 << len(context.attributes)):
        closure = full_mask
        for object_row in object_rows:
            if object_row & state == state:
                closure &= object_row
                row_intersections += 1
        masks.append(closure)
    return tuple(masks), row_intersections


def _enumerate_dg_masks(
    context: FormalContext,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], int, int, int]:
    """Enumerate exact closures, pseudo-intents, and lectic comparison counts."""

    closure_masks, row_intersections = _context_closure_masks(context)
    pseudo_intent_pairs, subset_comparisons, closure_comparisons = (
        _enumerate_pseudo_intents(closure_masks)
    )
    return (
        closure_masks,
        pseudo_intent_pairs,
        subset_comparisons,
        closure_comparisons,
        row_intersections,
    )


def _require_dg_canonical_carrier_fit(
    pseudo_intent_pairs: tuple[tuple[int, int], ...],
) -> int:
    """Reject exact bases beyond the canonical implication-system carrier."""

    count = len(pseudo_intent_pairs)
    if count > MAX_IMPLICATIONS:
        raise ValueError(
            "exact Duquenne-Guigues basis of "
            f"{count} implications exceeds the bounded canonical "
            f"implication-system carrier of {MAX_IMPLICATIONS} implications"
        )
    memberships = sum(closure.bit_count() for _, closure in pseudo_intent_pairs)
    if memberships > MAX_IMPLICATION_MEMBERSHIPS:
        raise ValueError(
            "exact Duquenne-Guigues basis of "
            f"{memberships} premise and conclusion memberships exceeds the "
            f"bounded canonical implication-system carrier of "
            f"{MAX_IMPLICATION_MEMBERSHIPS} memberships"
        )
    return memberships


def _reserved_dg_logical_work(
    context: FormalContext,
    states: int,
    closure_masks: tuple[int, ...],
    pseudo_intent_pairs: tuple[tuple[int, int], ...],
    implication_memberships: int,
    subset_comparisons: int,
    closure_comparisons: int,
    row_intersections: int,
) -> int:
    attribute_count = len(context.attributes)
    object_count = len(context.objects)
    incidence_count = len(context.incidence)
    return (
        # The plan's one closure-matrix and pseudo-intent scan, followed by
        # one canonical basis-equivalence pass. Each closure query needs at
        # most (attribute_count + 1) productive rounds over the retained
        # implication carrier.
        states * object_count
        + incidence_count
        + row_intersections
        + subset_comparisons
        + closure_comparisons
        + states
        * (attribute_count + 1)
        * (len(pseudo_intent_pairs) + implication_memberships)
        + attribute_count * states // 2
        + sum(mask.bit_count() for mask in closure_masks)
        + sum(
            state.bit_count() + closure.bit_count()
            for state, closure in pseudo_intent_pairs
        )
        + implication_memberships
    )


def _dg_output_reservation_payload(
    context: FormalContext,
    states: int,
    pseudo_intent_count: int,
    reserved_logical_work: int,
) -> dict[str, object]:
    attribute_count = len(context.attributes)
    full_subset = list(range(attribute_count))
    largest_state = states - 1
    closure_row = {
        "candidate_state": largest_state,
        "subset": full_subset,
        "closure": full_subset,
    }
    pseudo_intent = {
        "candidate_state": largest_state,
        "premise": full_subset,
        "closure": full_subset,
        "basis_implication_index": largest_state,
    }
    implication = {"premise": full_subset, "conclusion": full_subset}
    return {
        "context": context.model_dump(mode="json"),
        "source_attribute_indices": list(range(attribute_count)),
        "lectic_order": "BINARY_LECTIC_BY_MAXIMUM_DIFFERENCE",
        "closure_matrix": [closure_row for _ in range(states)],
        "pseudo_intents": [pseudo_intent for _ in range(pseudo_intent_count)],
        "basis": {
            "attributes": list(_basis_attribute_labels(attribute_count)),
            "implications": [implication for _ in range(pseudo_intent_count)],
        },
        "work": {
            "candidate_states": states,
            "context_closure_queries": states,
            "context_object_row_checks": states * len(context.objects),
            "context_incidence_loads": len(context.incidence),
            "context_row_intersections": states * len(context.objects),
            "pseudo_intent_subset_comparisons": states * states,
            "pseudo_intent_closure_comparisons": states * states,
            "basis_closure_queries": states,
            "basis_closure_work": MAX_DG_CANDIDATE_STATES
            * (attribute_count + 1)
            * (MAX_IMPLICATIONS + MAX_IMPLICATION_MEMBERSHIPS),
            "closure_matrix_memberships": 2 * states * attribute_count,
            "pseudo_intent_memberships": 2 * states * attribute_count,
            "implication_count": pseudo_intent_count,
            "implication_memberships": MAX_IMPLICATION_MEMBERSHIPS,
            "accounted_logical_work": reserved_logical_work,
            "reserved_logical_work": reserved_logical_work,
            "reserved_result_bytes": MAX_DG_RESULT_BYTES,
            "serialized_result_bytes": MAX_DG_RESULT_BYTES,
        },
    }


@dataclass(frozen=True)
class _DGBasisAdmissionPlan:
    """One admitted closure carrier and its exact execution reservations."""

    states: int
    closure_masks: tuple[int, ...]
    pseudo_intent_pairs: tuple[tuple[int, int], ...]
    subset_comparisons: int
    closure_comparisons: int
    row_intersections: int
    reserved_logical_work: int
    reserved_result_bytes: int


def _admit_duquenne_guigues_basis(context: FormalContext) -> _DGBasisAdmissionPlan:
    """Build the one semantic admission plan for a DG-basis invocation."""

    attribute_count = len(context.attributes)
    if attribute_count > MAX_DG_ATTRIBUTES:
        raise ValueError(
            "Duquenne-Guigues basis candidate-state domain exceeds the "
            "bounded conservative-fallback carrier of "
            f"{MAX_DG_CANDIDATE_STATES} candidate states "
            f"({MAX_DG_ATTRIBUTES} attributes)"
        )
    states = 1 << attribute_count
    (
        closure_masks,
        pseudo_intent_pairs,
        subset_comparisons,
        closure_comparisons,
        row_intersections,
    ) = _enumerate_dg_masks(context)
    implication_memberships = _require_dg_canonical_carrier_fit(pseudo_intent_pairs)
    reserved_logical_work = _reserved_dg_logical_work(
        context,
        states,
        closure_masks,
        pseudo_intent_pairs,
        implication_memberships,
        subset_comparisons,
        closure_comparisons,
        row_intersections,
    )
    if reserved_logical_work > MAX_DG_LOGICAL_WORK:
        raise ValueError(
            "Duquenne-Guigues basis closure-equivalence pass exceeds the bounded logical-work "
            f"limit of {MAX_DG_LOGICAL_WORK}"
        )

    payload = _dg_output_reservation_payload(
        context,
        states,
        len(pseudo_intent_pairs),
        reserved_logical_work,
    )
    try:
        reserved_result_bytes = len(
            encode_strict_json(
                payload,
                limits=CanonicalLimits(max_output_bytes=MAX_DG_RESULT_BYTES),
            )
        )
    except CanonicalizationError as exc:
        raise ValueError(
            "Duquenne-Guigues basis worst-case result exceeds the bounded "
            f"serialized-result limit of {MAX_DG_RESULT_BYTES} bytes"
        ) from exc
    return _DGBasisAdmissionPlan(
        states=states,
        closure_masks=closure_masks,
        pseudo_intent_pairs=pseudo_intent_pairs,
        subset_comparisons=subset_comparisons,
        closure_comparisons=closure_comparisons,
        row_intersections=row_intersections,
        reserved_logical_work=reserved_logical_work,
        reserved_result_bytes=reserved_result_bytes,
    )


class DGBasisClosureRow(StrictModel):
    """One source-index subset and its formal-context closure."""

    candidate_state: _DGCandidateState = Field(
        description=(
            "Binary subset mask: bit i is set exactly when source attribute i "
            "belongs to subset."
        )
    )
    subset: tuple[_DGAttributeIndex, ...] = Field(max_length=MAX_DG_ATTRIBUTES)
    closure: tuple[_DGAttributeIndex, ...] = Field(max_length=MAX_DG_ATTRIBUTES)


class PseudoIntent(StrictModel):
    """One pseudo-intent bound to its closure and canonical basis row."""

    candidate_state: _DGCandidateState = Field(
        description="Binary subset mask of the pseudo-intent premise."
    )
    premise: tuple[_DGAttributeIndex, ...] = Field(max_length=MAX_DG_ATTRIBUTES)
    closure: tuple[_DGAttributeIndex, ...] = Field(max_length=MAX_DG_ATTRIBUTES)
    basis_implication_index: StrictInt = Field(
        ge=0,
        lt=MAX_DG_CANDIDATE_STATES,
        description=(
            "Index of premise -> (closure minus premise) in basis.implications "
            "after canonical row normalization."
        ),
    )


class DGBasisWork(StrictModel):
    """Exact logical counts for one served request plus its reservations.

    Counts cover the one semantic admission plan consumed by the kernel and
    its one source-closure-equivalence pass. Independently supplied result
    data is checked by the explicit owner verifier, not ordinary result
    construction, so native and catalog invocations report identical work.
    """

    candidate_states: StrictInt = Field(ge=1, le=MAX_DG_CANDIDATE_STATES)
    context_closure_queries: StrictInt = Field(
        ge=1,
        le=MAX_DG_CANDIDATE_STATES,
    )
    context_object_row_checks: StrictInt = Field(
        ge=0,
        le=MAX_DG_CANDIDATE_STATES * MAX_OBJECTS,
    )
    context_incidence_loads: StrictInt = Field(
        ge=0,
        le=MAX_OBJECTS * MAX_DG_ATTRIBUTES,
    )
    context_row_intersections: StrictInt = Field(
        ge=0,
        le=MAX_DG_CANDIDATE_STATES * MAX_OBJECTS,
    )
    pseudo_intent_subset_comparisons: StrictInt = Field(
        ge=0,
        le=MAX_DG_CANDIDATE_STATES**2,
    )
    pseudo_intent_closure_comparisons: StrictInt = Field(
        ge=0,
        le=MAX_DG_CANDIDATE_STATES**2,
    )
    basis_closure_queries: StrictInt = Field(
        ge=1,
        le=MAX_DG_CANDIDATE_STATES,
    )
    basis_closure_work: StrictInt = Field(
        ge=0,
        le=(
            MAX_DG_CANDIDATE_STATES
            * (MAX_DG_ATTRIBUTES + 1)
            * (MAX_IMPLICATIONS + MAX_IMPLICATION_MEMBERSHIPS)
        ),
        description=(
            "Exact finite-implication closure-equivalence work performed by the "
            "producer after one semantic admission plan."
        ),
    )
    closure_matrix_memberships: StrictInt = Field(
        ge=0,
        le=2 * MAX_DG_CANDIDATE_STATES * MAX_DG_ATTRIBUTES,
    )
    pseudo_intent_memberships: StrictInt = Field(
        ge=0,
        le=2 * MAX_DG_CANDIDATE_STATES * MAX_DG_ATTRIBUTES,
    )
    implication_count: StrictInt = Field(ge=0, le=MAX_IMPLICATIONS)
    implication_memberships: StrictInt = Field(
        ge=0,
        le=MAX_IMPLICATION_MEMBERSHIPS,
    )
    accounted_logical_work: StrictInt = Field(ge=0, le=MAX_DG_LOGICAL_WORK)
    reserved_logical_work: StrictInt = Field(ge=0, le=MAX_DG_LOGICAL_WORK)
    reserved_result_bytes: StrictInt = Field(ge=1, le=MAX_DG_RESULT_BYTES)
    serialized_result_bytes: StrictInt = Field(ge=1, le=MAX_DG_RESULT_BYTES)

    @model_validator(mode="after")
    def bind_aggregate_work(self) -> Self:
        expected = (
            self.context_object_row_checks
            + self.context_incidence_loads
            + self.context_row_intersections
            + self.pseudo_intent_subset_comparisons
            + self.pseudo_intent_closure_comparisons
            + self.basis_closure_work
            + self.closure_matrix_memberships
            + self.pseudo_intent_memberships
            + self.implication_memberships
        )
        if self.accounted_logical_work != expected:
            raise ValueError(
                "accounted_logical_work does not match the exact component counts"
            )
        if self.accounted_logical_work > self.reserved_logical_work:
            raise ValueError("exact work exceeds the preflight logical-work reserve")
        if self.serialized_result_bytes > self.reserved_result_bytes:
            raise ValueError("serialized result exceeds its preflight byte reserve")
        return self


def _enumerate_pseudo_intents(
    closures: tuple[int, ...],
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pseudo_intents: list[tuple[int, int]] = []
    subset_comparisons = 0
    closure_comparisons = 0
    for state, closure in enumerate(closures):
        if state == closure:
            continue
        is_pseudo_intent = True
        for previous_state, previous_closure in pseudo_intents:
            subset_comparisons += 1
            if previous_state & state != previous_state:
                continue
            closure_comparisons += 1
            if previous_closure & state != previous_closure:
                is_pseudo_intent = False
                break
        if is_pseudo_intent:
            pseudo_intents.append((state, closure))
    return tuple(pseudo_intents), subset_comparisons, closure_comparisons


class CanonicalImplicationBasisResult(StrictModel):
    """The complete source-bound Duquenne--Guigues basis of a context."""

    context: FormalContext
    source_attribute_indices: tuple[_DGAttributeIndex, ...] = Field(
        max_length=MAX_DG_ATTRIBUTES,
        description=(
            "For each basis coordinate, the identical source-context attribute "
            "index. This explicit map binds the finite implication carrier to "
            "the retained FormalContext without narrowing source label syntax."
        ),
    )
    lectic_order: Literal["BINARY_LECTIC_BY_MAXIMUM_DIFFERENCE"] = Field(
        description=(
            "Candidate states increase by binary mask; equivalently, the largest "
            "differing source attribute belongs to the later subset."
        )
    )
    closure_matrix: tuple[DGBasisClosureRow, ...] = Field(
        max_length=MAX_DG_CANDIDATE_STATES,
        description=(
            "Every source-attribute subset and its context closure, in complete "
            "binary lectic state order."
        ),
    )
    pseudo_intents: tuple[PseudoIntent, ...] = Field(
        max_length=MAX_DG_CANDIDATE_STATES,
        description=(
            "Every pseudo-intent exactly once, in the candidate-state order in "
            "which its recursive defining condition is established."
        ),
    )
    basis: FiniteAttributeImplicationSystem = Field(
        description=(
            "The canonical FiniteAttributeImplicationSystem over source-index "
            "coordinate labels m0, m1, ...; source_attribute_indices binds that "
            "axis to context."
        )
    )
    work: DGBasisWork

    @classmethod
    def _from_kernel(cls, payload: dict[str, object]) -> Self:
        """Build a result emitted by the owner-local exhaustive kernel."""

        return cls.model_validate(payload)


__all__ = [
    "MAX_DG_ATTRIBUTES",
    "MAX_DG_CANDIDATE_STATES",
    "MAX_DG_LOGICAL_WORK",
    "MAX_DG_RESULT_BYTES",
    "CanonicalImplicationBasisResult",
    "DGBasisClosureRow",
    "DGBasisWork",
    "PseudoIntent",
]
