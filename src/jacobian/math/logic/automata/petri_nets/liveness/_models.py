"""Typed contracts for Petri-net transition liveness profiles."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.values import MAX_PETRI_TRANSITIONS

MAX_TRANSITION_LIVENESS_WORK = 1_000_000
MAX_TRANSITION_LIVENESS_OUTPUT_BYTES = 10 * 1024 * 1024


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(f"petri_net.transition_liveness.{reason}", message)


class TransitionLivenessEntry(StrictModel):
    """Liveness classification for one transition index.

    ``witness_state`` is a reachable marking from which the transition cannot
    fire in any continuation, and is present exactly for ``NOT_LIVE``.
    """

    transition: int = Field(ge=0, lt=MAX_PETRI_TRANSITIONS)
    status: Literal["LIVE", "NOT_LIVE", "UNKNOWN"]
    witness_state: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_witness_shape(self) -> Self:
        if (self.status == "NOT_LIVE") != (self.witness_state is not None):
            raise _validation_error(
                "witness_shape",
                "only a NOT_LIVE entry has a reachable witness state",
            )
        return self


class TransitionLivenessRequest(StrictModel):
    """Classify transitions using one bounded reachability graph."""

    source_graph: ReachabilityResult


class TransitionLivenessResult(StrictModel):
    """Transition liveness relative to a complete finite reachability graph.

    If the source graph is truncated, all statuses are UNKNOWN: represented
    edges cannot establish the universal condition over all reachable markings.
    """

    source_graph: ReachabilityResult
    transitions: tuple[TransitionLivenessEntry, ...] = Field(
        max_length=MAX_PETRI_TRANSITIONS
    )

    @model_validator(mode="after")
    def require_canonical_profile_shape(self) -> Self:
        if tuple(entry.transition for entry in self.transitions) != tuple(
            range(self.source_graph.net.transition_count)
        ):
            raise _validation_error(
                "transition_axis",
                "profile entries must cover the transition axis in order",
            )
        if self.source_graph.truncated:
            if any(
                entry.status != "UNKNOWN" or entry.witness_state is not None
                for entry in self.transitions
            ):
                raise _validation_error(
                    "incomplete_graph",
                    "a truncated graph supports only UNKNOWN statuses",
                )
        elif any(entry.status == "UNKNOWN" for entry in self.transitions):
            raise _validation_error(
                "complete_graph", "an untruncated graph must classify every transition"
            )
        if any(
            entry.witness_state is not None
            and entry.witness_state >= len(self.source_graph.states)
            for entry in self.transitions
        ):
            raise _validation_error(
                "witness_state", "witness states must lie on the source graph axis"
            )
        return self
