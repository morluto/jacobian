"""Domain-owned electrical-network operation adapters."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.math.electrical_networks import (
    effective_resistance,
    laplacian_matrix,
    node_potentials,
)
from jacobian.math.electrical_networks._models import (
    ConductanceNetwork,
    EffectiveResistanceRequest,
    EffectiveResistanceResult,
    LaplacianEntry,
    LaplacianRequest,
    LaplacianResult,
    NodePotentialRequest,
    NodePotentialResult,
    NodePotentialValue,
)


def _edge_triples(
    network: ConductanceNetwork,
) -> tuple[tuple[int, int, Fraction], ...]:
    return tuple(
        (edge.source, edge.target, edge.conductance.as_fraction())
        for edge in network.edges
    )


def compute_effective_resistance(
    request: EffectiveResistanceRequest,
) -> EffectiveResistanceResult:
    network = request.network
    value = effective_resistance(
        network.vertex_count,
        _edge_triples(network),
        request.terminal_a,
        request.terminal_b,
    )
    return EffectiveResistanceResult(
        effective_resistance=CanonicalRational.from_fraction(value),
        terminal_a=request.terminal_a,
        terminal_b=request.terminal_b,
    )


def compute_node_potentials(request: NodePotentialRequest) -> NodePotentialResult:
    network = request.network
    potentials = node_potentials(
        network.vertex_count,
        _edge_triples(network),
        request.source,
        request.sink,
    )
    values = tuple(
        NodePotentialValue(
            node=i,
            potential=CanonicalRational.from_fraction(potentials[i]),
        )
        for i in range(network.vertex_count)
    )
    return NodePotentialResult(
        source=request.source,
        sink=request.sink,
        potentials=values,
    )


def compute_laplacian(request: LaplacianRequest) -> LaplacianResult:
    network = request.network
    matrix = laplacian_matrix(network.vertex_count, _edge_triples(network))
    entries: list[LaplacianEntry] = []
    for row in range(network.vertex_count):
        for col in range(network.vertex_count):
            entries.append(
                LaplacianEntry(
                    row=row,
                    col=col,
                    value=CanonicalRational.from_fraction(matrix[row][col]),
                )
            )
    return LaplacianResult(
        vertex_count=network.vertex_count,
        entries=tuple(entries),
    )


__all__ = [
    "compute_effective_resistance",
    "compute_laplacian",
    "compute_node_potentials",
]
