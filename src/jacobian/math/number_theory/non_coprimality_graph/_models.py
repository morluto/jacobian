"""Typed contracts for the non-coprimality graph operation."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.math.graphs.values import MAX_GRAPH_LABEL_BYTES, SimpleUndirectedGraph

MAX_INTEGERS = 256
MAX_INTEGER_DIGITS = MAX_GRAPH_LABEL_BYTES

NonCoprimeInteger = Annotated[
    str,
    StringConstraints(
        pattern=rf"^[1-9][0-9]{{0,{MAX_INTEGER_DIGITS - 1}}}$",
        max_length=MAX_INTEGER_DIGITS,
        strict=True,
    ),
]


class NonCoprimalityGraphRequest(StrictModel):
    """Request to construct the non-coprimality graph of a set of integers."""

    integers: tuple[NonCoprimeInteger, ...] = Field(
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
