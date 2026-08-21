"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.petri_nets.values import (
    MAX_PETRI_MARKING,
    MAX_REACHABILITY_STATES,
    Marking,
    PetriNet,
    require_reachability_bounds,
)


class EnabledTransitionsRequest(StrictModel):
    """Find all enabled transitions at a marking."""

    net: PetriNet
    marking: Marking

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise ValueError("marking length must match place_count")
        return self


class EnabledTransitionsResult(StrictModel):
    """The set of enabled transition indices."""

    transitions: tuple[int, ...]


class FireTransitionRequest(StrictModel):
    """Fire one transition at a marking."""

    net: PetriNet
    marking: Marking
    transition: int = Field(ge=0)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.marking.tokens) != self.net.place_count:
            raise ValueError("marking length must match place_count")
        if not 0 <= self.transition < self.net.transition_count:
            raise ValueError("transition index out of range")
        return self


class FireTransitionResult(StrictModel):
    """Result of firing a transition."""

    status: Literal["FIRED", "NOT_ENABLED", "ESCAPES_DECLARED_ENVELOPE"]
    new_marking: Marking | None = None
    envelope_escape: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_consistent_outcome(self) -> Self:
        if self.status == "ESCAPES_DECLARED_ENVELOPE":
            if self.new_marking is not None or self.envelope_escape is None:
                raise ValueError(
                    "envelope escape must carry only the successor witness"
                )
            if any(token < 0 for token in self.envelope_escape):
                raise ValueError("envelope escape tokens must be nonnegative")
            if all(token <= MAX_PETRI_MARKING for token in self.envelope_escape):
                raise ValueError("envelope escape must contain an out-of-range token")
        elif self.new_marking is None or self.envelope_escape is not None:
            raise ValueError("ordinary firing outcomes must carry only a marking")
        return self


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix C = Post - Pre."""

    net: PetriNet


class IncidenceMatrixResult(StrictModel):
    """The incidence matrix."""

    incidence: tuple[tuple[int, ...], ...]


class ReachabilityRequest(StrictModel):
    """Compute the bounded reachability graph from an initial marking.

    The state count is admitted jointly with place and transition dimensions,
    bounding state cells, firing records, exploration work, and result bytes.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=MAX_REACHABILITY_STATES)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise ValueError("marking length must match place_count")
        require_reachability_bounds(self.net, self.max_states)
        return self


class ReachabilityFrontier(StrictModel):
    """One enabled firing omitted because its target is outside the state bound."""

    source_state: int = Field(ge=0)
    transition: int = Field(ge=0)
    target_marking: tuple[int, ...]


class ReachabilityEnvelopeEscape(ReachabilityFrontier):
    """First deterministic firing whose successor exceeds the marking domain."""

    @model_validator(mode="after")
    def require_outside_marking_envelope(self) -> Self:
        if not any(token > MAX_PETRI_MARKING for token in self.target_marking):
            raise ValueError("escape target must exceed the marking envelope")
        return self


class ReachabilityResult(StrictModel):
    """A complete graph, bounded prefix, or marking-envelope escape.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs. An envelope escape is
    a typed non-conclusion carrying the first deterministic firing witness.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(ge=1, le=MAX_REACHABILITY_STATES)
    states: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int], ...]
    status: Literal["COMPLETE", "TRUNCATED", "ESCAPES_DECLARED_ENVELOPE"]
    frontier: tuple[ReachabilityFrontier, ...]
    envelope_escape: ReachabilityEnvelopeEscape | None = None

    @model_validator(mode="after")
    def require_exact_bounded_graph(self) -> Self:
        from jacobian.math.petri_nets.operations import reachability_graph

        expected_states, expected_edges, expected_frontier, expected_escape = (
            reachability_graph(
                self.net,
                self.initial_marking,
                self.max_states,
            )
        )
        if self.states != tuple(expected_states):
            raise ValueError("states must equal the deterministic BFS states")
        if self.edges != tuple(expected_edges):
            raise ValueError("edges must equal the deterministic BFS edges")
        if self.frontier != tuple(
            ReachabilityFrontier(
                source_state=source,
                transition=transition,
                target_marking=target,
            )
            for source, transition, target in expected_frontier
        ):
            raise ValueError("frontier must equal the deterministic BFS frontier")
        expected_escape_value = (
            None
            if expected_escape is None
            else ReachabilityEnvelopeEscape(
                source_state=expected_escape[0],
                transition=expected_escape[1],
                target_marking=expected_escape[2],
            )
        )
        if self.envelope_escape != expected_escape_value:
            raise ValueError("envelope escape must equal the deterministic BFS witness")
        expected_status = (
            "ESCAPES_DECLARED_ENVELOPE"
            if expected_escape is not None
            else "TRUNCATED"
            if expected_frontier
            else "COMPLETE"
        )
        if self.status != expected_status:
            raise ValueError("status must agree with the deterministic BFS outcome")
        return self


__all__ = [
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "ReachabilityEnvelopeEscape",
    "ReachabilityFrontier",
    "ReachabilityRequest",
    "ReachabilityResult",
]
