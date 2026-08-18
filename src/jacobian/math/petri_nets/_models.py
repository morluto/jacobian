"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel
from jacobian.math.petri_nets.values import (
    Marking,
    PetriNet,
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

    fired: bool
    new_marking: tuple[int, ...] = Field(default=())


class IncidenceMatrixRequest(StrictModel):
    """Compute the incidence matrix C = Post - Pre."""

    net: PetriNet


class IncidenceMatrixResult(StrictModel):
    """The incidence matrix."""

    incidence: tuple[tuple[int, ...], ...]


class ReachabilityRequest(StrictModel):
    """Compute the bounded reachability graph from an initial marking.

    Bounds the state space to avoid unbounded exploration.
    """

    net: PetriNet
    initial_marking: Marking
    max_states: int = Field(default=10000, ge=1, le=100000)

    @model_validator(mode="after")
    def require_valid_marking_size(self) -> Self:
        if len(self.initial_marking.tokens) != self.net.place_count:
            raise ValueError("marking length must match place_count")
        return self


class ReachabilityFrontier(StrictModel):
    """One enabled firing omitted because its target is outside the state bound."""

    source_state: int = Field(ge=0)
    transition: int = Field(ge=0)
    target_marking: tuple[int, ...]


def _fired_marking(
    net: PetriNet,
    states: tuple[tuple[int, ...], ...],
    source: int,
    transition: int,
) -> tuple[int, ...]:
    if not 0 <= transition < net.transition_count:
        raise ValueError("transition index is out of range")
    marking = states[source]
    if any(
        marking[place] < net.pre[place][transition] for place in range(net.place_count)
    ):
        raise ValueError("reported firing is not enabled")
    return tuple(
        marking[place] - net.pre[place][transition] + net.post[place][transition]
        for place in range(net.place_count)
    )


def _require_valid_states(
    net: PetriNet,
    initial_marking: tuple[int, ...],
    max_states: int,
    states: tuple[tuple[int, ...], ...],
) -> None:
    if not states or states[0] != initial_marking:
        raise ValueError("states must begin with the initial marking")
    if len(states) > max_states:
        raise ValueError("states exceed max_states")
    if len(set(states)) != len(states):
        raise ValueError("states must be unique")
    if any(len(state) != net.place_count for state in states):
        raise ValueError("state marking length must match place_count")
    if any(token < 0 for state in states for token in state):
        raise ValueError("state markings must be non-negative")


def _expected_firings(
    net: PetriNet, states: tuple[tuple[int, ...], ...]
) -> set[tuple[int, int]]:
    return {
        (source, transition)
        for source, marking in enumerate(states)
        for transition in range(net.transition_count)
        if all(
            marking[place] >= net.pre[place][transition]
            for place in range(net.place_count)
        )
    }


def _validate_edges(
    net: PetriNet,
    states: tuple[tuple[int, ...], ...],
    edges: tuple[tuple[int, int, int], ...],
) -> tuple[set[tuple[int, int]], dict[int, set[int]]]:
    observed: set[tuple[int, int]] = set()
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(states))}
    for source, transition, target in edges:
        if not 0 <= source < len(states) or not 0 <= target < len(states):
            raise ValueError("edge state index is out of range")
        firing = (source, transition)
        if firing in observed:
            raise ValueError("each enabled firing must appear exactly once")
        observed.add(firing)
        if states[target] != _fired_marking(net, states, source, transition):
            raise ValueError("edge target does not match transition firing")
        adjacency[source].add(target)
    return observed, adjacency


def _validate_frontier(
    net: PetriNet,
    states: tuple[tuple[int, ...], ...],
    frontier: tuple[ReachabilityFrontier, ...],
    observed: set[tuple[int, int]],
) -> None:
    state_set = set(states)
    for record in frontier:
        if not 0 <= record.source_state < len(states):
            raise ValueError("frontier source_state is out of range")
        firing = (record.source_state, record.transition)
        if firing in observed:
            raise ValueError("each enabled firing must appear exactly once")
        observed.add(firing)
        if record.target_marking != _fired_marking(net, states, *firing):
            raise ValueError("frontier target does not match transition firing")
        if record.target_marking in state_set:
            raise ValueError("frontier target must be omitted from states")


def _require_reachable(adjacency: dict[int, set[int]]) -> None:
    reached = {0}
    pending = [0]
    while pending:
        for target in adjacency[pending.pop()]:
            if target not in reached:
                reached.add(target)
                pending.append(target)
    if reached != set(adjacency):
        raise ValueError("every reported state must be reachable from state 0")


class ReachabilityResult(StrictModel):
    """A complete graph or a bounded prefix with an explicit open frontier.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs.
    """

    net: PetriNet
    initial_marking: tuple[int, ...]
    max_states: int = Field(ge=1, le=100000)
    states: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int], ...]
    status: Literal["COMPLETE", "TRUNCATED"]
    frontier: tuple[ReachabilityFrontier, ...]

    @model_validator(mode="after")
    def require_exact_bounded_graph(self) -> Self:
        _require_valid_states(
            self.net, self.initial_marking, self.max_states, self.states
        )
        observed_firings, adjacency = _validate_edges(self.net, self.states, self.edges)
        _validate_frontier(self.net, self.states, self.frontier, observed_firings)
        expected_firings = _expected_firings(self.net, self.states)
        if observed_firings != expected_firings:
            raise ValueError("edges and frontier must cover every enabled firing")
        if self.frontier and len(self.states) != self.max_states:
            raise ValueError("a nonempty frontier requires an exhausted state bound")
        expected_status = "TRUNCATED" if self.frontier else "COMPLETE"
        if self.status != expected_status:
            raise ValueError("status must agree with the open frontier")
        _require_reachable(adjacency)
        return self


__all__ = [
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "ReachabilityFrontier",
    "ReachabilityRequest",
    "ReachabilityResult",
]
