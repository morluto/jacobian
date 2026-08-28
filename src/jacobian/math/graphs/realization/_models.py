"""Typed wire contracts for graph realization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from jacobian._models import StrictModel
from jacobian.math.graphs.values import IndexedSimpleUndirectedGraph

# ---------------------------------------------------------------------------
# Degree sequences
# ---------------------------------------------------------------------------

MAX_GRAPH_LENGTH = 64
MAX_GRAPH_DEGREE = 63


class DegreeSequence(StrictModel):
    """A sequence of nonnegative vertex degrees."""

    degrees: tuple[int, ...] = Field(min_length=1, max_length=MAX_GRAPH_LENGTH)

    @model_validator(mode="after")
    def require_valid_degrees(self) -> Self:
        if any(d < 0 for d in self.degrees):
            raise PydanticCustomError(
                "graph.degrees_must_be_nonnegative", "degrees must be nonnegative"
            )
        if any(d > MAX_GRAPH_DEGREE for d in self.degrees):
            raise PydanticCustomError(
                "graph.degrees_must_not_exceed_the_maximum_degree_bound",
                "degrees must not exceed the maximum degree bound",
            )
        return self


# ---------------------------------------------------------------------------
# is_graphical operation
# ---------------------------------------------------------------------------


class DegreeSequenceRequest(StrictModel):
    sequence: DegreeSequence


class DegreeSequenceResult(StrictModel):
    is_graphical: bool
    degree_sum: int = Field(ge=0)
    vertex_count: int = Field(ge=1)


# ---------------------------------------------------------------------------
# realization (construct a graph) operation
# ---------------------------------------------------------------------------


class GraphRealizationRequest(StrictModel):
    sequence: DegreeSequence


class GraphRealizationResult(StrictModel):
    is_graphical: bool
    vertex_count: int = Field(ge=1)
    edges: tuple[tuple[int, int], ...] = Field(default=())


# ---------------------------------------------------------------------------
# graphicality check (with certificate) operation
# ---------------------------------------------------------------------------


class GraphicalityCheckRequest(StrictModel):
    sequence: DegreeSequence


class GraphicalityCheckResult(StrictModel):
    is_graphical: bool
    degree_sum: int = Field(ge=0)
    vertex_count: int = Field(ge=1)
    certificate: str = ""


# ---------------------------------------------------------------------------
# check operation
# ---------------------------------------------------------------------------


class RealizationCheckRequest(StrictModel):
    sequence: DegreeSequence
    graph: IndexedSimpleUndirectedGraph

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.sequence.degrees) != self.graph.vertex_count:
            raise PydanticCustomError(
                "graph.sequence_length_must_match_graph_vertex_count",
                "sequence length must match graph vertex_count",
            )
        return self


class RealizationCheckResult(StrictModel):
    is_realization: bool
    expected_degrees: tuple[int, ...]
    actual_degrees: tuple[int, ...]


__all__ = [
    "MAX_GRAPH_DEGREE",
    "MAX_GRAPH_LENGTH",
    "DegreeSequence",
    "DegreeSequenceRequest",
    "DegreeSequenceResult",
    "GraphRealizationRequest",
    "GraphRealizationResult",
    "GraphicalityCheckRequest",
    "GraphicalityCheckResult",
    "RealizationCheckRequest",
    "RealizationCheckResult",
]
