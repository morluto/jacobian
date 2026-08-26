"""Provider-independent values for exact Petri net operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel

MAX_PETRI_PLACES = 64
MAX_PETRI_TRANSITIONS = 64
MAX_PETRI_MARKING = 1000
MAX_PETRI_ARC_WEIGHT = 1000
MAX_REACHABILITY_STATES = 100_000
MAX_REACHABILITY_STATE_TOKEN_CELLS = 100_000
MAX_REACHABILITY_FIRING_RECORDS = 100_000
MAX_REACHABILITY_EXPLORATION_WORK = 1_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"petri_net.{reason}", message)


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
            raise _validation_error("pre_row_count", "pre must have place_count rows")
        if len(self.post) != self.place_count:
            raise _validation_error("post_row_count", "post must have place_count rows")
        for row in self.pre:
            if len(row) != self.transition_count:
                raise _validation_error(
                    "pre_row_width", "pre row must have transition_count entries"
                )
            if any(w < 0 for w in row):
                raise _validation_error(
                    "pre_weight_sign", "pre weights must be non-negative"
                )
            if any(w > MAX_PETRI_ARC_WEIGHT for w in row):
                raise _validation_error(
                    "pre_weight_bound",
                    f"pre weights must not exceed {MAX_PETRI_ARC_WEIGHT}",
                )
        for row in self.post:
            if len(row) != self.transition_count:
                raise _validation_error(
                    "post_row_width", "post row must have transition_count entries"
                )
            if any(w < 0 for w in row):
                raise _validation_error(
                    "post_weight_sign", "post weights must be non-negative"
                )
            if any(w > MAX_PETRI_ARC_WEIGHT for w in row):
                raise _validation_error(
                    "post_weight_bound",
                    f"post weights must not exceed {MAX_PETRI_ARC_WEIGHT}",
                )
        return self


class Marking(StrictModel):
    """A marking (token assignment) of a Petri net."""

    tokens: tuple[int, ...]

    @model_validator(mode="after")
    def require_valid_marking(self) -> Self:
        if any(t < 0 for t in self.tokens):
            raise _validation_error(
                "marking_token_sign", "marking tokens must be non-negative"
            )
        if any(t > MAX_PETRI_MARKING for t in self.tokens):
            raise _validation_error(
                "marking_token_bound",
                f"marking tokens must not exceed {MAX_PETRI_MARKING}",
            )
        return self


class FiringSequence(StrictModel):
    """A sequence of transition firings."""

    transitions: tuple[int, ...] = Field(default=())


def require_reachability_bounds(net: PetriNet, max_states: int) -> None:
    """Admit BFS work jointly with state, place, and transition dimensions."""
    state_cells = max_states * net.place_count
    firing_records = max_states * net.transition_count
    exploration_work = 2 * firing_records * net.place_count
    if state_cells > MAX_REACHABILITY_STATE_TOKEN_CELLS:
        raise ValueError("reachability state-token cells exceed the work bound")
    if firing_records > MAX_REACHABILITY_FIRING_RECORDS:
        raise ValueError("reachability firing records exceed the work bound")
    if exploration_work > MAX_REACHABILITY_EXPLORATION_WORK:
        raise ValueError("reachability exploration exceeds the work bound")


__all__ = [
    "MAX_PETRI_ARC_WEIGHT",
    "MAX_PETRI_MARKING",
    "MAX_PETRI_PLACES",
    "MAX_PETRI_TRANSITIONS",
    "MAX_REACHABILITY_EXPLORATION_WORK",
    "MAX_REACHABILITY_FIRING_RECORDS",
    "MAX_REACHABILITY_STATES",
    "MAX_REACHABILITY_STATE_TOKEN_CELLS",
    "FiringSequence",
    "Marking",
    "PetriNet",
    "require_reachability_bounds",
]
