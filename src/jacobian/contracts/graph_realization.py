"""Typed wire contracts for graph realization operations."""

from __future__ import annotations

from typing import Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel

# ---------------------------------------------------------------------------
# Degree sequences
# ---------------------------------------------------------------------------

MAX_GRAPH_LENGTH = 64
MAX_GRAPH_DEGREE = 63


class DegreeSequence(ContractModel):
    """A sequence of nonnegative vertex degrees."""

    degrees: tuple[int, ...] = Field(min_length=1, max_length=MAX_GRAPH_LENGTH)

    @model_validator(mode="after")
    def require_valid_degrees(self) -> Self:
        if any(d < 0 for d in self.degrees):
            raise ValueError("degrees must be nonnegative")
        if any(d > MAX_GRAPH_DEGREE for d in self.degrees):
            raise ValueError("degrees must not exceed the maximum degree bound")
        return self


class GraphEdges(ContractModel):
    """A simple undirected graph for the check operation."""

    vertex_count: int = Field(ge=1, le=MAX_GRAPH_LENGTH)
    edges: tuple[tuple[int, int], ...] = Field(
        default=(),
        max_length=MAX_GRAPH_LENGTH * (MAX_GRAPH_LENGTH - 1) // 2,
    )

    @model_validator(mode="after")
    def require_valid_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for source, target in self.edges:
            if not (
                0 <= source < self.vertex_count and 0 <= target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            if source == target:
                raise ValueError("self-loops are not allowed")
            canonical = (min(source, target), max(source, target))
            if canonical in seen:
                raise ValueError("edges must be unique")
            seen.add(canonical)
        return self


# ---------------------------------------------------------------------------
# is_graphical operation
# ---------------------------------------------------------------------------


class DegreeSequenceRequest(ContractModel):
    sequence: DegreeSequence


class DegreeSequenceResult(ContractModel):
    is_graphical: bool
    degree_sum: int = Field(ge=0)
    vertex_count: int = Field(ge=1)


# ---------------------------------------------------------------------------
# realization (construct a graph) operation
# ---------------------------------------------------------------------------


class GraphRealizationRequest(ContractModel):
    sequence: DegreeSequence


class GraphRealizationResult(ContractModel):
    is_graphical: bool
    vertex_count: int = Field(ge=1)
    edges: tuple[tuple[int, int], ...] = Field(default=())
    convention: str = "NETWORKX_HAVEL_HAKIMI"


# ---------------------------------------------------------------------------
# graphicality check (with certificate) operation
# ---------------------------------------------------------------------------


class GraphicalityCheckRequest(ContractModel):
    sequence: DegreeSequence


class GraphicalityCheckResult(ContractModel):
    is_graphical: bool
    degree_sum: int = Field(ge=0)
    vertex_count: int = Field(ge=1)
    certificate: str = ""


# ---------------------------------------------------------------------------
# check operation
# ---------------------------------------------------------------------------


class RealizationCheckRequest(ContractModel):
    sequence: DegreeSequence
    graph: GraphEdges

    @model_validator(mode="after")
    def require_matching_lengths(self) -> Self:
        if len(self.sequence.degrees) != self.graph.vertex_count:
            raise ValueError("sequence length must match graph vertex_count")
        return self


class RealizationCheckResult(ContractModel):
    is_realization: bool
    expected_degrees: tuple[int, ...]
    actual_degrees: tuple[int, ...]
    convention: str = "NETWORKX_DEGREE"


__all__ = [
    "MAX_GRAPH_DEGREE",
    "MAX_GRAPH_LENGTH",
    "DegreeSequence",
    "DegreeSequenceRequest",
    "DegreeSequenceResult",
    "GraphEdges",
    "GraphRealizationRequest",
    "GraphRealizationResult",
    "GraphicalityCheckRequest",
    "GraphicalityCheckResult",
    "RealizationCheckRequest",
    "RealizationCheckResult",
]
