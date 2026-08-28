"""Typed wire contracts for exact electrical-network operations."""

from __future__ import annotations

from pydantic import Field

from jacobian._exact import CanonicalRational
from jacobian._models import StrictModel

MAX_NETWORK_VERTICES = 128
MAX_NETWORK_EDGES = 512

# Each conductance is reduced and has numerator and denominator at most this many
# decimal digits. Results (effective resistance, node potentials, Laplacian
# entries) are ratios of degree-at-most-127 weighted spanning-forest/tree
# polynomials: after clearing the common denominator, each component is bounded
# by MAX_NETWORK_EDGES * MAX_CONDUCTANCE_DIGITS + log10(128**126) digits
# (512 * 50 + 266 = 25,866), comfortably inside the canonical 32,768-digit
# rational ceiling.
MAX_CONDUCTANCE_DIGITS = 50


class ConductanceEdge(StrictModel):
    """One undirected edge with a positive rational conductance (1/resistance)."""

    source: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    target: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    conductance: CanonicalRational


class ConductanceNetwork(StrictModel):
    """An undirected graph of positive conductances over vertices 0..vertex_count-1."""

    vertex_count: int = Field(ge=2, le=MAX_NETWORK_VERTICES)
    edges: tuple[ConductanceEdge, ...] = Field(
        min_length=1, max_length=MAX_NETWORK_EDGES
    )


class EffectiveResistanceRequest(StrictModel):
    """Effective resistance between two distinct terminals of a conductance network."""

    network: ConductanceNetwork
    terminal_a: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    terminal_b: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)


class EffectiveResistanceResult(StrictModel):
    """Exact effective resistance between two terminals."""

    effective_resistance: CanonicalRational
    terminal_a: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    terminal_b: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)


class NodePotentialRequest(StrictModel):
    """Solve the Dirichlet problem: inject 1 unit of current at source, extract at sink."""

    network: ConductanceNetwork
    source: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)
    sink: int = Field(ge=0, le=MAX_NETWORK_VERTICES - 1)


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
