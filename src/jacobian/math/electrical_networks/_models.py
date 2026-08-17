"""Typed wire contracts for exact electrical-network operations."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian._models import StrictModel

MAX_NETWORK_VERTICES = 64
MAX_NETWORK_EDGES = 512

# Each conductance is reduced and has numerator and denominator at most this many
# decimal digits. Results (effective resistance, node potentials, Laplacian
# entries) are ratios of degree-at-most-63 weighted spanning-forest/tree
# polynomials: after clearing the common denominator, each component is bounded
# by MAX_NETWORK_EDGES * MAX_CONDUCTANCE_DIGITS + log10(64**62) digits
# (512 * 50 + 112 = 25,712), comfortably inside the canonical 32,768-digit
# rational ceiling.
MAX_CONDUCTANCE_DIGITS = 50


class ConductanceEdge(StrictModel):
    """One undirected edge with a positive rational conductance (1/resistance)."""

    source: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    conductance: CanonicalRational

    @model_validator(mode="after")
    def require_distinct_positive(self) -> Self:
        if self.source == self.target:
            raise ValueError("edge endpoint must be distinct")
        if self.conductance.as_fraction() <= 0:
            raise ValueError("conductance must be strictly positive")
        require_bounded_rational(
            self.conductance,
            max_digits=MAX_CONDUCTANCE_DIGITS,
            label="conductance",
        )
        return self


class ConductanceNetwork(StrictModel):
    """An undirected graph of positive conductances over vertices 0..vertex_count-1."""

    vertex_count: int = Field(ge=2, le=MAX_NETWORK_VERTICES)
    edges: tuple[ConductanceEdge, ...] = Field(
        min_length=1, max_length=MAX_NETWORK_EDGES
    )

    @model_validator(mode="after")
    def require_valid_unique_edges(self) -> Self:
        seen: set[tuple[int, int]] = set()
        for edge in self.edges:
            if not (
                0 <= edge.source < self.vertex_count
                and 0 <= edge.target < self.vertex_count
            ):
                raise ValueError("edge vertices must be in 0..vertex_count-1")
            key = (
                (edge.source, edge.target)
                if edge.source < edge.target
                else (edge.target, edge.source)
            )
            if key in seen:
                raise ValueError("edges must be unique (ignoring direction)")
            seen.add(key)
        return self


def _require_connected(network: ConductanceNetwork) -> None:
    """Reject networks whose reduced Laplacian solve is singular."""

    adjacency: list[list[int]] = [[] for _ in range(network.vertex_count)]
    for edge in network.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)

    seen: list[bool] = [False] * network.vertex_count
    seen[0] = True
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if not seen[neighbor]:
                seen[neighbor] = True
                stack.append(neighbor)

    if not all(seen):
        raise ValueError("network must be connected")


class EffectiveResistanceRequest(StrictModel):
    """Effective resistance between two distinct terminals of a conductance network."""

    network: ConductanceNetwork
    terminal_a: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    terminal_b: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)

    @model_validator(mode="after")
    def require_valid_distinct_terminals(self) -> Self:
        if not (0 <= self.terminal_a < self.network.vertex_count):
            raise ValueError("terminal_a must be in 0..vertex_count-1")
        if not (0 <= self.terminal_b < self.network.vertex_count):
            raise ValueError("terminal_b must be in 0..vertex_count-1")
        if self.terminal_a == self.terminal_b:
            raise ValueError("terminals must be distinct")
        return self

    @model_validator(mode="after")
    def require_connected_network(self) -> Self:
        _require_connected(self.network)
        return self


class EffectiveResistanceResult(StrictModel):
    """Exact effective resistance between two terminals."""

    effective_resistance: CanonicalRational
    terminal_a: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    terminal_b: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    method: Literal["SYMPY_REDUCED_LAPLACIAN_SOLVE"] = "SYMPY_REDUCED_LAPLACIAN_SOLVE"


class NodePotentialRequest(StrictModel):
    """Solve the Dirichlet problem: inject 1 unit of current at source, extract at sink."""

    network: ConductanceNetwork
    source: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    sink: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)

    @model_validator(mode="after")
    def require_valid_distinct_terminals(self) -> Self:
        if not (0 <= self.source < self.network.vertex_count):
            raise ValueError("source must be in 0..vertex_count-1")
        if not (0 <= self.sink < self.network.vertex_count):
            raise ValueError("sink must be in 0..vertex_count-1")
        if self.source == self.sink:
            raise ValueError("source and sink must be distinct")
        return self

    @model_validator(mode="after")
    def require_connected_network(self) -> Self:
        _require_connected(self.network)
        return self


class NodePotentialValue(StrictModel):
    """One node's exact potential after solving a Dirichlet problem."""

    node: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    potential: CanonicalRational


class NodePotentialResult(StrictModel):
    """Exact node potentials for unit current injection."""

    source: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    sink: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    potentials: tuple[NodePotentialValue, ...] = Field(
        min_length=2, max_length=MAX_NETWORK_VERTICES
    )
    method: Literal["SYMPY_LAPLACIAN_SOLVE"] = "SYMPY_LAPLACIAN_SOLVE"


class LaplacianEntry(StrictModel):
    """One entry of the exact conductance-weighted Laplacian matrix."""

    row: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    col: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    value: CanonicalRational


class LaplacianRequest(StrictModel):
    """Compute the conductance-weighted Laplacian matrix of a network."""

    network: ConductanceNetwork


class LaplacianResult(StrictModel):
    """Exact Laplacian matrix as a flat list of (row, col, value) entries."""

    vertex_count: int = Field(ge=2, le=MAX_NETWORK_VERTICES)
    entries: tuple[LaplacianEntry, ...] = Field(min_length=1)
    method: Literal["SYMPY_LAPLACIAN"] = "SYMPY_LAPLACIAN"


__all__ = [
    "ConductanceEdge",
    "ConductanceNetwork",
    "EffectiveResistanceRequest",
    "EffectiveResistanceResult",
    "LaplacianEntry",
    "LaplacianRequest",
    "LaplacianResult",
    "NodePotentialRequest",
    "NodePotentialResult",
    "NodePotentialValue",
]
