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
                raise ValueError("envelope escape must carry only the successor")
            if all(token <= MAX_PETRI_MARKING for token in self.envelope_escape):
                raise ValueError("envelope escape must exceed the marking bound")
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

    Bounds the state space to avoid unbounded exploration.
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


class ReachabilityResult(StrictModel):
    """The bounded reachability graph.

    Each state is a marking tuple. The graph is a mapping from marking
    to a list of (transition, resulting_marking) pairs.
    """

    states: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int, int], ...]
    truncated: bool


class SiphonTrapRequest(StrictModel):
    """Check for siphons and traps in a Petri net."""

    net: PetriNet

    @model_validator(mode="after")
    def require_bounded_places(self) -> Self:
        if self.net.place_count > 20:
            raise ValueError(
                "siphon/trap check supports at most 20 places for exact enumeration"
            )
        return self


class SiphonTrapResult(StrictModel):
    """Minimal siphons and traps of the net.

    Each siphon/trap is represented as a tuple of place indices.
    """

    siphons: tuple[tuple[int, ...], ...]
    traps: tuple[tuple[int, ...], ...]


__all__ = [
    "EnabledTransitionsRequest",
    "EnabledTransitionsResult",
    "FireTransitionRequest",
    "FireTransitionResult",
    "IncidenceMatrixRequest",
    "IncidenceMatrixResult",
    "ReachabilityRequest",
    "ReachabilityResult",
    "SiphonTrapRequest",
    "SiphonTrapResult",
]
