"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from typing import Literal, Self

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


__all__ = [
    "MAX_SIPHON_TRAP_WORK",
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "PetriMarkingState",
    "PetriPlaceSubset",
    "PetriReachabilityEdge",
    "ReachabilityRequest",
    "ReachabilityResult",
    "SiphonTrapRequest",
    "SiphonTrapResult",
]
