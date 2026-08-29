"""Typed contracts for the non-coprimality graph operation."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_INTEGERS = 256
MAX_INTEGER_DIGITS = 256


class NonCoprimalityGraphRequest(StrictModel):
    """Request to construct the non-coprimality graph of a set of integers."""

    integers: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_INTEGERS,
    )

class NonCoprimalityGraphResult(StrictModel):
    """The non-coprimality graph of a set of integers."""

    integers: tuple[CanonicalInteger, ...]
    graph: SimpleUndirectedGraph

__all__ = [
    "MAX_INTEGERS",
    "MAX_INTEGER_DIGITS",
    "NonCoprimalityGraphRequest",
    "NonCoprimalityGraphResult",
]
