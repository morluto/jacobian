"""Contracts for finite Petri-net reachability profiles."""

from typing import Any, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.logic.automata.petri_nets._models import ReachabilityResult
from jacobian.math.logic.automata.petri_nets.values import MAX_PETRI_PLACES

MAX_REACHABILITY_TOKEN_PROFILE_OUTPUT_BYTES = 10 * 1024 * 1024
MAX_REACHABILITY_TOKEN_PROFILE_WORK = 1_000_000


def _validation_error(reason: str, message: str) -> PydanticCustomError:
    return PydanticCustomError(
        f"petri_net.reachability_token_profile.{reason}", message
    )


class ReachabilityTokenRange(StrictModel):
    """Extrema for one place, with lowest-index source-state witnesses."""

    place: int = Field(ge=0, lt=MAX_PETRI_PLACES)
    minimum: int = Field(ge=0)
    minimum_state: int = Field(ge=0)
    maximum: int = Field(ge=0)
    maximum_state: int = Field(ge=0)

    @model_validator(mode="after")
    def require_ordered_extrema(self) -> Self:
        if self.minimum > self.maximum:
            raise _validation_error("extrema", "minimum must not exceed maximum")
        return self


class ReachabilityTokenProfileRequest(StrictModel):
    """Summarize token extrema on an admitted finite reachability graph."""

    source_graph: ReachabilityResult


class ReachabilityTokenProfileResult(StrictModel):
    """Exact token extrema over represented states, complete iff graph is closed."""

    source_graph: ReachabilityResult
    completeness: Literal["COMPLETE", "OBSERVED_PREFIX"]
    place_ranges: tuple[ReachabilityTokenRange, ...] = Field(
        max_length=MAX_PETRI_PLACES
    )
    total_minimum: int = Field(ge=0)
    total_minimum_state: int = Field(ge=0)
    total_maximum: int = Field(ge=0)
    total_maximum_state: int = Field(ge=0)

    @model_validator(mode="after")
    def require_profile_axes(self) -> Self:
        state_count = len(self.source_graph.states)
        if self.completeness != (
            "OBSERVED_PREFIX" if self.source_graph.truncated else "COMPLETE"
        ):
            raise _validation_error(
                "completeness", "profile completeness must match the source graph"
            )
        if tuple(item.place for item in self.place_ranges) != tuple(
            range(self.source_graph.net.place_count)
        ):
            raise _validation_error(
                "place_axis", "place ranges must cover the source place axis"
            )
        witnesses = (
            self.total_minimum_state,
            self.total_maximum_state,
            *(item.minimum_state for item in self.place_ranges),
            *(item.maximum_state for item in self.place_ranges),
        )
        if any(state >= state_count for state in witnesses):
            raise _validation_error(
                "state_axis", "extremum witnesses must lie on the source state axis"
            )
        return self

    @classmethod
    def _from_kernel(cls, **values: Any) -> "ReachabilityTokenProfileResult":
        """Build from the admitted source-graph scan without replaying extrema."""
        return cls.model_construct(**values)
