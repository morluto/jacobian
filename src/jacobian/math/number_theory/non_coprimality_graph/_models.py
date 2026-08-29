"""Typed contracts for the non-coprimality graph operation."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_INTEGERS = 256


class NonCoprimalityGraphRequest(StrictModel):
    """Request to construct the non-coprimality graph of a set of integers."""

    integers: tuple[int, ...] = Field(
        min_length=1,
        max_length=MAX_INTEGERS,
    )

    @model_validator(mode="after")
    def validate_integers(self) -> Self:
        for v in self.integers:
            if v <= 0:
                raise PydanticCustomError(
                    "non_coprimality.must_be_positive",
                    "all integers must be positive",
                )
        if len(set(self.integers)) != len(self.integers):
            raise PydanticCustomError(
                "non_coprimality.must_be_distinct",
                "integers must be distinct",
            )
        return self


class NonCoprimalityGraphResult(StrictModel):
    """The non-coprimality graph of a set of integers."""

    integers: tuple[int, ...]
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_INTEGERS",
    "NonCoprimalityGraphRequest",
    "NonCoprimalityGraphResult",
]
