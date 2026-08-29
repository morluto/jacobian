"""Typed contracts for the non-coprimality graph operation."""

from __future__ import annotations

from math import gcd
from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._exact import CanonicalInteger
from jacobian._models import StrictModel
from jacobian.canonical import CanonicalLimits, encode_strict_json
from jacobian.math.graphs.values import SimpleUndirectedGraph

MAX_INTEGERS = 256
MAX_INTEGER_DIGITS = 256


def _validate_integer_source(integers: tuple[str, ...]) -> None:
    if not 1 <= len(integers) <= MAX_INTEGERS:
        raise PydanticCustomError(
            "non_coprimality.size",
            f"integers must contain between 1 and {MAX_INTEGERS} values",
        )
    values: list[int] = []
    for index, value in enumerate(integers):
        if len(value.lstrip("-")) > MAX_INTEGER_DIGITS:
            raise PydanticCustomError(
                "non_coprimality.digits",
                f"integer {index} exceeds the {MAX_INTEGER_DIGITS}-digit bound",
            )
        parsed = int(value)
        if parsed <= 0:
            raise PydanticCustomError(
                "non_coprimality.must_be_positive",
                "all integers must be positive",
            )
        values.append(parsed)
    if len(set(values)) != len(values):
        raise PydanticCustomError(
            "non_coprimality.must_be_distinct",
            "integers must be distinct",
        )

    labels = tuple(sorted(integers, key=int))
    all_edges = [[left, right] for left in labels for right in labels if left < right]
    payload = {
        "integers": list(labels),
        "graph": {"vertices": list(labels), "edges": all_edges},
    }
    if len(encode_strict_json(payload)) > CanonicalLimits().max_output_bytes:
        raise PydanticCustomError(
            "non_coprimality.result_too_large",
            "the graph result exceeds the canonical output limit",
        )


class NonCoprimalityGraphRequest(StrictModel):
    """Request to construct the non-coprimality graph of a set of integers."""

    integers: tuple[CanonicalInteger, ...] = Field(
        min_length=1,
        max_length=MAX_INTEGERS,
    )

    @model_validator(mode="after")
    def validate_integers(self) -> Self:
        _validate_integer_source(self.integers)
        return self


class NonCoprimalityGraphResult(StrictModel):
    """The non-coprimality graph of a set of integers."""

    integers: tuple[CanonicalInteger, ...]
    graph: SimpleUndirectedGraph

    @model_validator(mode="after")
    def require_source_graph_contract(self) -> Self:
        _validate_integer_source(self.integers)
        expected_vertices = tuple(sorted(self.integers, key=int))
        if self.graph.vertices != expected_vertices:
            raise PydanticCustomError(
                "non_coprimality.graph_source_mismatch",
                "graph vertices must be the source integers in numeric order",
            )
        values = {label: int(label) for label in self.integers}
        expected_edges = {
            tuple(sorted((left, right)))
            for left in self.integers
            for right in self.integers
            if left < right and gcd(values[left], values[right]) > 1
        }
        if set(self.graph.edges) != expected_edges:
            raise PydanticCustomError(
                "non_coprimality.graph_edges_mismatch",
                "graph edges must be exactly the non-coprime source pairs",
            )
        return self


__all__ = [
    "MAX_INTEGERS",
    "MAX_INTEGER_DIGITS",
    "NonCoprimalityGraphRequest",
    "NonCoprimalityGraphResult",
    "_validate_integer_source",
]
