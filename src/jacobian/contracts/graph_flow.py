"""Typed wire contracts for graph flow and cut operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel
from jacobian.contracts.exact import CanonicalRational


class CapacitatedEdge(ContractModel):
    """One directed edge with a rational capacity."""

    source: int = Field(ge=0, le=63)
    target: int = Field(ge=0, le=63)
    capacity: CanonicalRational


class FlowGraph(ContractModel):
    """A directed capacitated graph for flow problems."""

    vertex_count: int = Field(ge=2, le=64)
    edges: tuple[CapacitatedEdge, ...] = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def require_valid_vertices(self) -> Self:
        for edge in self.edges:
            if not (0 <= edge.source < self.vertex_count and 0 <= edge.target < self.vertex_count):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
        return self


class MaxFlowRequest(ContractModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)


class MaxFlowResult(ContractModel):
    flow_value: CanonicalRational
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)
    convention: Literal["NETWORKX_MAXIMUM_FLOW"] = "NETWORKX_MAXIMUM_FLOW"


class MinCutRequest(ContractModel):
    graph: FlowGraph
    source: int = Field(ge=0, le=63)
    sink: int = Field(ge=0, le=63)


class MinCutResult(ContractModel):
    cut_value: CanonicalRational
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    convention: Literal["NETWORKX_MINIMUM_CUT"] = "NETWORKX_MINIMUM_CUT"
