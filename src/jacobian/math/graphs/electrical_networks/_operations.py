"""Domain-owned electrical-network operation adapters."""

from __future__ import annotations

from fractions import Fraction

from jacobian._exact import CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.graphs.electrical_networks import (
    effective_resistance,
    laplacian_matrix,
    node_potentials,
)
from jacobian.math.graphs.electrical_networks._models import (
    MAX_CONDUCTANCE_DIGITS,
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


def _domain_error(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location,
        code=f"electrical_network.{code}",
        message=message,
    )


def _admit_network(network: ConductanceNetwork) -> None:
    """Check mathematical graph and conductance preconditions once per call."""
    seen: set[tuple[int, int]] = set()
    for index, edge in enumerate(network.edges):
        location = ("network", "edges", index)
        if edge.source == edge.target:
            _domain_error(
                (*location, "source"),
                "edge_endpoints_not_distinct",
                "edge endpoint must be distinct",
            )
        if edge.conductance.as_fraction() <= 0:
            _domain_error(
                (*location, "conductance"),
                "conductance_not_positive",
                "conductance must be strictly positive",
            )
        try:
            from jacobian._exact import require_bounded_rational

            require_bounded_rational(
                edge.conductance,
                max_digits=MAX_CONDUCTANCE_DIGITS,
                label="conductance",
            )
        except ValueError as exc:
            _domain_error(
                (*location, "conductance"),
                "conductance_exceeds_digit_bound",
                str(exc),
            )
        if not (
            0 <= edge.source < network.vertex_count
            and 0 <= edge.target < network.vertex_count
        ):
            _domain_error(
                location,
                "edge_vertex_out_of_range",
                "edge vertices must be in 0..vertex_count-1",
            )
        key = (min(edge.source, edge.target), max(edge.source, edge.target))
        if key in seen:
            _domain_error(
                location,
                "duplicate_edges",
                "edges must be unique (ignoring direction)",
            )
        seen.add(key)


def _require_connected(network: ConductanceNetwork) -> None:
    adjacency: list[list[int]] = [[] for _ in range(network.vertex_count)]
    for edge in network.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    if len(seen) != network.vertex_count:
        _domain_error(
            ("network",), "network_not_connected", "network must be connected"
        )


def _admit_terminals(
    network: ConductanceNetwork,
    first: int,
    second: int,
    names: tuple[str, str],
) -> None:
    for name, value in zip(names, (first, second), strict=True):
        if not 0 <= value < network.vertex_count:
            _domain_error(
                (name,),
                f"{name}_out_of_range",
                f"{name} must be in 0..vertex_count-1",
            )
    if first == second:
        _domain_error(
            names,
            "terminals_not_distinct"
            if names == ("terminal_a", "terminal_b")
            else "source_sink_not_distinct",
            "terminals must be distinct"
            if names == ("terminal_a", "terminal_b")
            else "source and sink must be distinct",
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
    _admit_network(network)
    _admit_terminals(
        network, request.terminal_a, request.terminal_b, ("terminal_a", "terminal_b")
    )
    _require_connected(network)
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
    _admit_network(network)
    _admit_terminals(network, request.source, request.sink, ("source", "sink"))
    _require_connected(network)
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
    _admit_network(network)
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
