"""Typed wire contracts for bounded directed-graph operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian.contracts.base import ContractModel


class DirectedArc(ContractModel):
    """One directed edge (arc) with a tail and a head vertex."""

    tail: int = Field(ge=0, le=255)
    head: int = Field(ge=0, le=255)


class DirectedGraph(ContractModel):
    """A finite simple directed graph allowing loops.

    Vertices are the integer range 0..vertex_count-1.  Arcs are ordered
    pairs (tail, head); loops are permitted because they are meaningful
    cycle witnesses.  Parallel duplicate arcs are rejected.
    """

    vertex_count: int = Field(ge=1, le=256)
    arcs: tuple[tuple[int, int], ...] = Field(max_length=4096)

    @model_validator(mode="after")
    def require_valid_arcs(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for tail, head in self.arcs:
            if not (0 <= tail < self.vertex_count and 0 <= head < self.vertex_count):
                raise ValueError("arc vertices must be in 0..vertex_count-1")
            pair = (tail, head)
            if pair in seen:
                raise ValueError("directed arcs must be unique")
            seen.add(pair)
        return self


class ReachabilityRequest(ContractModel):
    """Compute reachability from a single source vertex."""

    graph: DirectedGraph
    source: int = Field(ge=0, le=255)
    target: int | None = Field(default=None, ge=0, le=255)

    @model_validator(mode="after")
    def require_source_in_range(self) -> Self:
        if not (0 <= self.source < self.graph.vertex_count):
            raise ValueError("source must be in 0..vertex_count-1")
        if self.target is not None and not (0 <= self.target < self.graph.vertex_count):
            raise ValueError("target must be in 0..vertex_count-1")
        return self


class ReachabilityResult(ContractModel):
    """Reachability ledger from a single source."""

    source: int
    reachable: tuple[int, ...]
    unreachable: tuple[int, ...]
    distances: dict[int, int]
    predecessors: dict[int, int | None]
    target: int | None = None
    target_reachable: bool | None = None
    shortest_path: tuple[int, ...] | None = None


class StrongComponentsRequest(ContractModel):
    """Compute strongly connected components and condensation DAG."""

    graph: DirectedGraph


class StrongComponentsResult(ContractModel):
    """SCC partition and condensation DAG."""

    component_count: int
    component_ids: dict[int, int]
    components: dict[int, tuple[int, ...]]
    condensation_arcs: tuple[tuple[int, int], ...]
    source_components: tuple[int, ...]
    sink_components: tuple[int, ...]
    is_strongly_connected: bool


class AcyclicOrderRequest(ContractModel):
    """Compute a topological order or detect a directed cycle."""

    graph: DirectedGraph


class AcyclicOrderResult(ContractModel):
    """Closed acyclic-or-cyclic result."""

    status: Literal["ACYCLIC", "CYCLIC"]
    topological_order: tuple[int, ...] | None = None
    positions: dict[int, int] | None = None
    cycle_witness: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_consistent_payload(self) -> Self:
        if self.status == "ACYCLIC":
            if self.topological_order is None or self.positions is None:
                raise ValueError("ACYCLIC requires topological_order and positions")
            if self.cycle_witness is not None:
                raise ValueError("ACYCLIC must not include a cycle_witness")
        elif self.status == "CYCLIC":
            if self.cycle_witness is None:
                raise ValueError("CYCLIC requires a cycle_witness")
            if self.topological_order is not None or self.positions is not None:
                raise ValueError(
                    "CYCLIC must not include topological_order or positions"
                )
        return self


class TransitiveClosureRequest(ContractModel):
    """Compute the transitive closure of a directed graph.

    The reflexive convention is explicit so the caller knows whether
    (v, v) pairs are included even when no nonempty cycle through v exists.
    """

    graph: DirectedGraph
    reflexive: bool = False


class TransitiveClosureResult(ContractModel):
    """Transitive closure relation and closure graph."""

    closure_pairs: tuple[tuple[int, int], ...]
    vertex_count: int
    reflexive: bool


class DegreeProfileRequest(ContractModel):
    """Compute in-degree and out-degree for every vertex."""

    graph: DirectedGraph


class DegreeProfileResult(ContractModel):
    """Exact degree profile with sources, sinks, and isolated vertices."""

    in_degrees: tuple[int, ...]
    out_degrees: tuple[int, ...]
    sources: tuple[int, ...]
    sinks: tuple[int, ...]
    isolated: tuple[int, ...]
