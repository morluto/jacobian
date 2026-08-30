"""Typed contracts for the uniform-subset intersection graph operation."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_UNIFORM_SUBSET_ITEMS = 100_000


def _uniform_subset_admission_error(n: int, k: int) -> tuple[str, str] | None:
    if n < 0 or k < 0 or k > n:
        return ("k_out_of_range", "k must satisfy 0 <= k <= n")
    vertex_count = _bounded_binomial(n, k, 256)
    if vertex_count > 256:
        return (
            "vertex_count_exceeded",
            "uniform-subset family exceeds the 256-vertex graph carrier",
        )
    item_count = vertex_count * k
    if item_count > MAX_UNIFORM_SUBSET_ITEMS:
        return (
            "materialization_work_exceeded",
            "uniform-subset family exceeds the subset-materialization work bound",
        )
    label_bytes = vertex_count * (4 + k * (len(str(max(1, n))) + 1))
    if label_bytes > CanonicalLimits().max_output_bytes:
        return (
            "output_bound_exceeded",
            "uniform-subset labels exceed the canonical output budget",
        )
    return None


def _bounded_binomial(n: int, k: int, limit: int) -> int:
    """Return ``binomial(n, k)`` or ``limit + 1`` without oversized integers."""
    count = 1
    for index in range(1, min(k, n - k) + 1):
        count = count * (n - index + 1) // index
        if count > limit:
            return limit + 1
    return count


class UniformSubsetIntersectionRequest(StrictModel):
    """Request to construct a uniform-subset intersection graph."""

    n: int = Field(ge=0)
    k: int = Field(ge=0)
    threshold: int
    relation: Literal[
        "INTERSECTION_LT_THRESHOLD",
        "INTERSECTION_EQ_THRESHOLD",
        "INTERSECTION_GT_THRESHOLD",
    ]

    @model_validator(mode="after")
    def require_bounded_uniform_family(self) -> Self:
        failure = _uniform_subset_admission_error(self.n, self.k)
        if failure is not None:
            code, message = failure
            raise PydanticCustomError(f"uniform_subset.{code}", message)
        return self


class UniformSubsetIntersectionResult(StrictModel):
    """The uniform-subset intersection graph."""

    n: int
    k: int
    threshold: int
    relation: str
    graph: SimpleUndirectedGraph


__all__ = [
    "MAX_UNIFORM_SUBSET_ITEMS",
    "UniformSubsetIntersectionRequest",
    "UniformSubsetIntersectionResult",
    "_uniform_subset_admission_error",
]
