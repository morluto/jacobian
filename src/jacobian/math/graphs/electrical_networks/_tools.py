"""Electrical-network operation declarations."""

from fractions import Fraction
from typing import Any

from jacobian._exact import CanonicalRational, require_bounded_rational
from jacobian.catalog.models import (
    MathTool,
    OperationDomainValidationError,
    OperationExample,
)
from jacobian.math._rational_height import RationalHeight, sum_heights
from jacobian.math.graphs.electrical_networks import operations as native
from jacobian.math.graphs.electrical_networks._models import (
    MAX_CONDUCTANCE_DIGITS,
    MAX_NETWORK_SOLVE_WORK,
    MAX_NETWORK_VERTICES,
    ConductanceNetwork,
    EffectiveResistanceRequest,
    EffectiveResistanceResult,
    LaplacianEntry,
    LaplacianNetwork,
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


def _admit_network(network: ConductanceNetwork | LaplacianNetwork) -> None:
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


def _accumulated_coefficient_height(network: ConductanceNetwork) -> int:
    """Bound Laplacian diagonal height from incident conductance sums.

    Off-diagonal entries copy one conductance. Each diagonal is the sum of the
    incident conductances, so distinct denominators accumulate before FLINT
    constructs or solves the reduced system.
    """

    incident: list[list[RationalHeight]] = [[] for _ in range(network.vertex_count)]
    for edge in network.edges:
        height = RationalHeight.from_canonical(edge.conductance)
        incident[edge.source].append(height)
        incident[edge.target].append(height)
    tallest = 1
    for heights in incident:
        if not heights:
            continue
        combined = sum_heights(heights)
        tallest = max(
            tallest,
            combined.numerator_digits,
            combined.denominator_digits,
        )
    return tallest


def _admit_solve(network: ConductanceNetwork) -> None:
    dimension = network.vertex_count - 1
    coefficient_height = _accumulated_coefficient_height(network)
    scalar_work = dimension**3 * coefficient_height
    if scalar_work > MAX_NETWORK_SOLVE_WORK:
        _domain_error(
            ("network",),
            "solve_work_bound",
            "reduced Laplacian exceeds the exact solve-work bound",
        )
    result_digits = len(network.edges) * MAX_CONDUCTANCE_DIGITS + max(
        0, network.vertex_count - 2
    ) * len(str(MAX_NETWORK_VERTICES))
    if result_digits > 32_768:
        _domain_error(
            ("network",),
            "solve_result_bound",
            "reduced Laplacian solution exceeds the canonical rational bound",
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
    network: ConductanceNetwork | LaplacianNetwork,
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
    _admit_solve(network)
    value = native.effective_resistance(
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
    _admit_solve(network)
    potentials = native.node_potentials(
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
    matrix = native.laplacian_matrix(network.vertex_count, _edge_triples(network))
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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="electrical_network.effective_resistance.compute",
        title="Compute the exact effective resistance between two terminals",
        description="Compute the exact rational effective resistance between two terminals of an undirected conductance network by solving the reduced Laplacian system over QQ.",
        request_type=EffectiveResistanceRequest,
        result_type=EffectiveResistanceResult,
        run=compute_effective_resistance,
        tags=("graph", "electrical-network", "effective-resistance", "exact"),
        examples=(
            OperationExample(
                name="triangle_equal_resistances",
                description="Effective resistance of two vertices in a triangle with unit resistances.",
                input={
                    "network": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 0,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                    "terminal_a": 0,
                    "terminal_b": 1,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="electrical_network.node_potentials.compute",
        title="Compute exact node potentials for unit current injection",
        description="Solve the Dirichlet problem: inject 1 ampere at source and extract 1 ampere at sink, returning exact rational node potentials with the sink gauge fixed at zero.",
        request_type=NodePotentialRequest,
        result_type=NodePotentialResult,
        run=compute_node_potentials,
        tags=("graph", "electrical-network", "node-potential", "exact"),
        examples=(
            OperationExample(
                name="path_of_two_edges",
                description="Node potentials for a path graph of 3 vertices with unit conductances.",
                input={
                    "network": {
                        "vertex_count": 3,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                            {
                                "source": 1,
                                "target": 2,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                    "source": 0,
                    "sink": 2,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="electrical_network.laplacian.compute",
        title="Compute the exact conductance-weighted Laplacian matrix",
        description="Build the exact rational conductance-weighted graph Laplacian of an undirected network, returned as a flat list of (row, col, value) entries.",
        request_type=LaplacianRequest,
        result_type=LaplacianResult,
        run=compute_laplacian,
        tags=("graph", "electrical-network", "laplacian", "exact"),
        examples=(
            OperationExample(
                name="single_edge",
                description="Laplacian of a two-vertex network with one unit-conductance edge.",
                input={
                    "network": {
                        "vertex_count": 2,
                        "edges": [
                            {
                                "source": 0,
                                "target": 1,
                                "conductance": {"num": "1", "den": "1"},
                            },
                        ],
                    },
                },
            ),
        ),
    ),
)


__all__ = ["TOOLS"]
