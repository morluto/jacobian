"""Typed contracts for the eventual hitting profile operation."""

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_STATES = 32


class EventualHittingProfileRequest(StrictModel):
    """Request for the eventual hitting probability profile."""

    matrix: tuple[tuple[CanonicalRational, ...], ...]
    target_states: tuple[int, ...]


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
