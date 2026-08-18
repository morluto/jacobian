"""Typed wire contracts for Petri net operations."""

from __future__ import annotations

from typing import Self

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


class ReachabilityResult(StrictModel):
    """The bounded reachability graph.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs.
    """

    states: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int], ...]
    truncated: bool


__all__ = [
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "ReachabilityRequest",
    "ReachabilityResult",
]
