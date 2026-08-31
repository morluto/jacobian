"""Typed contracts for the eventual hitting profile operation."""

from typing import Annotated, Self

from pydantic import Field, StrictInt, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

# Structural safety limit: the matrix dimension must be representable as a
# single JSON array and the linear solve must be feasible.  The actual
# work-based admission (matrix-digit height, transient-system growth, and
# transport-byte budget) lives in ``operations.py`` and is the authority for
# whether a given profile is admissible.
MAX_STATES = 4096


class EventualHittingProfileRequest(StrictModel):
    """Request for the eventual hitting probability profile."""

    matrix: tuple[
        Annotated[tuple[CanonicalRational, ...], Field(max_length=MAX_STATES)], ...
    ] = Field(min_length=1, max_length=MAX_STATES)
    target_states: tuple[StrictInt, ...] = Field(
        min_length=1,
        max_length=MAX_STATES,
        description=(
            "Strictly increasing target-state indices in the matrix range "
            "0..dimension-1."
        ),
    )

    @model_validator(mode="after")
    def require_structural_request(self) -> Self:
        dimension = len(self.matrix)
        if any(len(row) != dimension for row in self.matrix):
            raise PydanticCustomError(
                "markov_chain.eventual_hitting_matrix_not_square",
                "matrix must be square",
            )
        if tuple(sorted(set(self.target_states))) != self.target_states:
            raise PydanticCustomError(
                "markov_chain.eventual_hitting_targets_not_canonical",
                "target_states must be strictly increasing",
            )
        if any(state < 0 or state >= dimension for state in self.target_states):
            raise PydanticCustomError(
                "markov_chain.eventual_hitting_target_out_of_range",
                "target_states must refer to matrix rows",
            )
        return self


class EventualHittingProfileResult(StrictModel):
    """The complete eventual hitting probability profile."""

    matrix: tuple[tuple[CanonicalRational, ...], ...]
    target_states: tuple[int, ...]
    hitting_probabilities: tuple[CanonicalRational, ...]
    zero_states: tuple[int, ...]
    proper_states: tuple[int, ...]
    almost_sure_states: tuple[int, ...]


__all__ = [
    "MAX_STATES",
    "EventualHittingProfileRequest",
    "EventualHittingProfileResult",
]
