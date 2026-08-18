"""Provider-independent values for exact Petri net operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian._models import StrictModel

MAX_PETRI_PLACES = 64
MAX_PETRI_TRANSITIONS = 64
MAX_PETRI_MARKING = 1000
MAX_PETRI_ARC_WEIGHT = 1000


class PetriNet(StrictModel):
    """A weighted place/transition Petri net.

    The net has ``place_count`` places and ``transition_count`` transitions.
    Arcs are specified by ``pre[p][t]`` (pre-condition) and ``post[p][t]``
    (post-condition) non-negative integer matrices.
    """

    place_count: int = Field(ge=1, le=MAX_PETRI_PLACES)
    transition_count: int = Field(ge=1, le=MAX_PETRI_TRANSITIONS)
    pre: tuple[tuple[int, ...], ...]
    post: tuple[tuple[int, ...], ...]

    @model_validator(mode="after")
    def require_valid_matrices(self) -> Self:
        if len(self.pre) != self.place_count:
            raise ValueError("pre must have place_count rows")
        if len(self.post) != self.place_count:
            raise ValueError("post must have place_count rows")
        for row in self.pre:
            if len(row) != self.transition_count:
                raise ValueError("pre row must have transition_count entries")
            if any(w < 0 for w in row):
                raise ValueError("pre weights must be non-negative")
        for row in self.post:
            if len(row) != self.transition_count:
                raise ValueError("post row must have transition_count entries")
            if any(w < 0 for w in row):
                raise ValueError("post weights must be non-negative")
        return self


class Marking(StrictModel):
    """A marking (token assignment) of a Petri net."""

    tokens: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_marking(self) -> Self:
        if any(t < 0 for t in self.tokens):
            raise ValueError("marking tokens must be non-negative")
        return self


class FiringSequence(StrictModel):
    """A sequence of transition firings."""

    transitions: tuple[int, ...] = Field(default=())


__all__ = [
    "MAX_PETRI_ARC_WEIGHT",
    "MAX_PETRI_MARKING",
    "MAX_PETRI_PLACES",
    "MAX_PETRI_TRANSITIONS",
    "FiringSequence",
    "Marking",
    "PetriNet",
]
