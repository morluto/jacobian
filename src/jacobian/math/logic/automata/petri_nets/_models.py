"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.petri_nets.values import (
    MAX_PETRI_MARKING,
    MAX_REACHABILITY_STATES,
    Marking,
    PetriMarkingState,
    PetriNet,
    PetriPlaceSubset,
    PetriReachabilityEdge,
)
from jacobian.math.matrices.values import IntegerMatrix

MAX_SIPHON_TRAP_WORK = 20_000_000
MAX_SIPHON_TRAP_PLACES = 20
MAX_FIRING_SEQUENCE_LENGTH = 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"petri_net.{reason}", message)


class EnabledTransitionsRequest(StrictModel):
    """Find all enabled transitions at a marking."""

    net: PetriNet
    marking: Marking

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class EnabledTransitionsResult(StrictModel):
    """The set of enabled transition indices bound to the source marking."""

    net: PetriNet
    marking: Marking
    transitions: tuple[int, ...]

    @model_validator(mode="after")
    def require_source_shape(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if self.transitions != tuple(sorted(set(self.transitions))) or any(
            not 0 <= transition < self.net.transition_count
            for transition in self.transitions
        ):
            raise _validation_error("transition_axis", "transitions must be canonical")
        return self


class FireTransitionRequest(StrictModel):
    """Fire one transition at a marking."""

    net: PetriNet
    marking: Marking
    transition: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if not 0 <= self.transition < self.net.transition_count:
            raise _validation_error("transition_index", "transition index out of range")
        return self


class FireTransitionResult(StrictModel):
    """Result of firing a transition, retaining its source context."""

    net: PetriNet
    marking: Marking
    transition: int = Field(ge=0)
    status: Literal["FIRED", "NOT_ENABLED", "ESCAPES_DECLARED_ENVELOPE"]
    new_marking: Marking | None = None
    envelope_escape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if not 0 <= self.transition < self.net.transition_count:
            raise _validation_error("transition_index", "transition index out of range")
        if self.status == "ESCAPES_DECLARED_ENVELOPE":
            if self.new_marking is not None or self.envelope_escape is None:
                raise _validation_error(
                    "escape_payload", "envelope escape must carry only the successor"
                )
            if all(token <= MAX_PETRI_MARKING for token in self.envelope_escape):
                raise _validation_error(
                    "escape_bound", "envelope escape must exceed the marking bound"
                )
        elif self.new_marking is None or self.envelope_escape is not None:
            raise _validation_error(
                "ordinary_payload", "ordinary firing outcomes must carry only a marking"
            )
        if (
            self.new_marking is not None
            and len(self.new_marking.tokens) != self.net.place_count
        ):
            raise _validation_error(
                "new_marking_length", "new marking length must match place_count"
            )
        if (
            self.envelope_escape is not None
            and len(self.envelope_escape) != self.net.place_count
        ):
            raise _validation_error(
                "escape_length", "envelope escape length must match place_count"
            )
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix C = Post - Pre."""

    net: PetriNet


class IncidenceMatrixResult(StrictModel):
    """The incidence matrix bound to its net's place/transition axes."""

    net: PetriNet
    incidence: IntegerMatrix

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if (
            self.incidence.row_count != self.net.place_count
            or self.incidence.column_count != self.net.transition_count
        ):
            raise _validation_error(
                "incidence_axes", "incidence axes must match the net"
            )
        return self


class ReachabilityRequest(StrictModel):
    """Compute the bounded reachability graph from an initial marking.

    Bounds the state space to avoid unbounded exploration.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=MAX_REACHABILITY_STATES)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        return self


class ReachabilityResult(StrictModel):
    """The bounded reachability graph.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    states: tuple[PetriMarkingState, ...]
    edges: tuple[PetriReachabilityEdge, ...]
    truncated: bool

    @model_validator(mode="after")
    def require_source_axes(self) -> Self:
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if tuple(state.state_index for state in self.states) != tuple(
            range(len(self.states))
        ):
            raise _validation_error(
                "state_axis", "states must use a complete index axis"
            )
        if any(
            state.place_axis != tuple(range(self.net.place_count))
            or len(state.marking.tokens) != self.net.place_count
            for state in self.states
        ):
            raise _validation_error(
                "state_marking_length", "state markings must match place_count"
            )
        if any(
            edge.source_state >= len(self.states)
            or edge.target_state >= len(self.states)
            or edge.transition >= self.net.transition_count
            for edge in self.edges
        ):
            raise _validation_error(
                "edge_axis", "reachability edges must use declared axes"
            )
        return self


class SiphonTrapRequest(StrictModel):
    """Check for siphons and traps in a Petri net."""

    net: PetriNet


class SiphonTrapResult(StrictModel):
    """Minimal siphons and traps of the net.

    Each siphon/trap is represented as a tuple of place indices.
    """

    net: PetriNet
    siphons: tuple[PetriPlaceSubset, ...]
    traps: tuple[PetriPlaceSubset, ...]

    @model_validator(mode="after")
    def require_place_axes(self) -> Self:
        if any(
            place >= self.net.place_count
            for subset in (*self.siphons, *self.traps)
            for place in subset.places
        ):
            raise _validation_error("place_axis", "subsets must use the net place axis")
        return self


class FiringSequenceReplayRequest(StrictModel):
    """Replay a bounded transition sequence from a source marking."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(
        default=(),
        max_length=MAX_FIRING_SEQUENCE_LENGTH,
        description=(
            "Ordered transition indices to fire from the source marking; "
            "a later transition cannot repair an earlier disabled one."
        ),
    )

    @model_validator(mode="after")
    def require_valid_sequence_axes(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if any(
            not 0 <= transition < self.net.transition_count
            for transition in self.sequence
        ):
            raise _validation_error(
                "transition_axis", "sequence transitions must use the net axis"
            )
        return self


class FiringSequenceReplayResult(StrictModel):
    """Closed firing-sequence replay: FIRES or first-blocked BLOCKED."""

    net: PetriNet
    marking: Marking
    sequence: tuple[int, ...] = Field(max_length=MAX_FIRING_SEQUENCE_LENGTH)
    status: Literal["FIRES", "BLOCKED"]
    prefix_markings: tuple[Marking, ...]
    final_marking: Marking | None = None
    parikh: tuple[int, ...]
    state_equation_residual: tuple[int, ...]
    blocked_index: int | None = None
    deficit: tuple[int, ...] | None = None
    first_deficient_place: int | None = None

    def _require_shared_axes(self) -> None:
        if len(self.marking.tokens) != self.net.place_count:
            raise _validation_error(
                "marking_length", "marking length must match place_count"
            )
        if any(
            not 0 <= transition < self.net.transition_count
            for transition in self.sequence
        ):
            raise _validation_error(
                "transition_axis", "sequence transitions must use the net axis"
            )
        if len(self.parikh) != self.net.transition_count or any(
            count < 0 for count in self.parikh
        ):
            raise _validation_error(
                "parikh_axis", "parikh must be a nonnegative transition multiset"
            )
        if len(self.state_equation_residual) != self.net.place_count:
            raise _validation_error(
                "residual_axis", "state-equation residual must use the place axis"
            )

    def _require_fires_payload(self) -> None:
        assert self.status == "FIRES"
        if (
            self.final_marking is None
            or self.blocked_index is not None
            or self.deficit is not None
            or self.first_deficient_place is not None
        ):
            raise _validation_error(
                "fires_payload", "FIRES must carry only a final marking"
            )
        if len(self.prefix_markings) != len(self.sequence):
            raise _validation_error(
                "fires_prefix", "FIRES must record one marking per transition"
            )
        if sum(self.parikh) != len(self.sequence):
            raise _validation_error(
                "fires_parikh", "FIRES parikh must count every transition"
            )
        if any(entry != 0 for entry in self.state_equation_residual):
            raise _validation_error(
                "fires_residual", "FIRES residual must witness state-equation equality"
            )
        if self.sequence and self.prefix_markings[-1] != self.final_marking:
            raise _validation_error(
                "fires_final", "FIRES final marking must extend the prefix ledger"
            )
        if not self.sequence and self.final_marking != self.marking:
            raise _validation_error(
                "fires_empty", "an empty sequence leaves the marking unchanged"
            )

    def _require_blocked_payload(self) -> None:
        assert self.status == "BLOCKED"
        if (
            self.final_marking is not None
            or self.blocked_index is None
            or self.deficit is None
            or self.first_deficient_place is None
        ):
            raise _validation_error(
                "blocked_payload", "BLOCKED must carry only its obstruction"
            )
        if not 0 <= self.blocked_index < len(self.sequence):
            raise _validation_error(
                "blocked_index", "blocked index must point into the sequence"
            )
        if len(self.prefix_markings) != self.blocked_index:
            raise _validation_error(
                "blocked_prefix", "BLOCKED must record exactly the fired prefix"
            )
        if sum(self.parikh) != self.blocked_index:
            raise _validation_error(
                "blocked_parikh", "BLOCKED parikh must count the fired prefix"
            )
        if len(self.deficit) != self.net.place_count or not any(
            entry > 0 for entry in self.deficit
        ):
            raise _validation_error(
                "blocked_deficit", "BLOCKED deficit must be a nonempty place profile"
            )
        if self.deficit[self.first_deficient_place] <= 0 or any(
            entry > 0 for entry in self.deficit[: self.first_deficient_place]
        ):
            raise _validation_error(
                "blocked_first",
                "first deficient place must be the first positive deficit",
            )

    @model_validator(mode="after")
    def require_branch_consistency(self) -> Self:
        self._require_shared_axes()
        if self.status == "FIRES":
            self._require_fires_payload()
        else:
            self._require_blocked_payload()
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


class PetriInvariantsRequest(StrictModel):
    """Compute P-invariants and T-invariants as exact integer modules."""

    net: PetriNet


class PetriInvariantsResult(PetriInvariantsRequest):
    """P/T-invariant bases with their incidence rank profile.

    Deserialization checks only the canonical shape: incidence axes match
    the net, every vector uses its axis, each basis is sign-normalized,
    sorted, and unique with the nullity the Smith rank predicts, and the
    replay flag is set. The owner-local kernel establishes that every
    vector lies in the incidence (left) kernel.
    """

    incidence: IntegerMatrix
    incidence_rank: int = Field(ge=0)
    p_invariants: tuple[tuple[int, ...], ...] = Field(default=())
    t_invariants: tuple[tuple[int, ...], ...] = Field(default=())
    replayed: bool

    @model_validator(mode="after")
    def require_canonical_invariant_shape(self) -> Self:
        if (
            self.incidence.row_count != self.net.place_count
            or self.incidence.column_count != self.net.transition_count
        ):
            raise _validation_error(
                "invariants_incidence_axes", "incidence axes must match the net"
            )
        if self.incidence_rank > min(self.net.place_count, self.net.transition_count):
            raise _validation_error(
                "invariants_rank_bound",
                "the incidence rank fits inside the net axes",
            )
        if len(self.p_invariants) != self.net.place_count - self.incidence_rank:
            raise _validation_error(
                "invariants_p_nullity",
                "the P-basis size must match the left nullity",
            )
        if len(self.t_invariants) != self.net.transition_count - self.incidence_rank:
            raise _validation_error(
                "invariants_t_nullity",
                "the T-basis size must match the right nullity",
            )
        for vectors, ambient in (
            (self.p_invariants, self.net.place_count),
            (self.t_invariants, self.net.transition_count),
        ):
            if vectors != tuple(sorted(set(vectors))):
                raise _validation_error(
                    "invariants_not_canonical",
                    "invariant bases must be sorted and unique",
                )
            for vector in vectors:
                if len(vector) != ambient or not any(vector):
                    raise _validation_error(
                        "invariants_axis",
                        "invariant vectors must use their axis nontrivially",
                    )
                if next(value for value in vector if value) < 0:
                    raise _validation_error(
                        "invariants_sign",
                        "invariant vectors must be sign-normalized",
                    )
        if not self.replayed:
            raise _validation_error(
                "invariants_not_replayed",
                "an invariant result must replay its bases in the kernel",
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> Self:
        return cls.model_construct(**values)


__all__ = [
    "MAX_FIRING_SEQUENCE_LENGTH",
    "MAX_SIPHON_TRAP_WORK",
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "FiringSequenceReplayRequest",
    "FiringSequenceReplayResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "PetriInvariantsRequest",
    "PetriInvariantsResult",
    "PetriMarkingState",
    "PetriPlaceSubset",
    "PetriReachabilityEdge",
    "ReachabilityRequest",
    "ReachabilityResult",
    "SiphonTrapRequest",
    "SiphonTrapResult",
]
