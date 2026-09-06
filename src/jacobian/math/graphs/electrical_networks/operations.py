"""Exact electrical-network kernels with private maintained backends."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from jacobian.math.graphs.electrical_networks._models import (
    EffectiveResistanceResult,
    LaplacianResult,
    NodePotentialResult,
)

__all__ = [
    "effective_resistance",
    "laplacian_matrix",
    "node_potentials",
    "verify_effective_resistance",
    "verify_laplacian",
    "verify_node_potentials",
]


def _laplacian(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
) -> Any:
    """Build the conductance-weighted Laplacian as a SymPy Matrix of rationals."""

    from sympy import Matrix, Rational

    matrix = Matrix.zeros(vertex_count, vertex_count)
    for source, target, conductance in edges:
        g = Rational(conductance.numerator, conductance.denominator)
        matrix[source, source] += g
        matrix[target, target] += g
        matrix[source, target] -= g
        matrix[target, source] -= g
    return matrix


def laplacian_matrix(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
) -> list[list[Fraction]]:
    """Return the exact Laplacian as a list-of-lists of Fractions."""

    lap = _laplacian(vertex_count, edges)
    rows: list[list[Fraction]] = []
    for row in range(vertex_count):
        entries: list[Fraction] = []
        for col in range(vertex_count):
            val = lap[row, col]
            entries.append(Fraction(int(val.p), int(val.q)))
        rows.append(entries)
    return rows


def effective_resistance(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
    terminal_a: int,
    terminal_b: int,
) -> Fraction:
    """Compute exact effective resistance by solving the reduced Laplacian system.

    Fix one node's potential as a gauge, solve the invertible reduced system
    ``L_reduced x = e_a - e_b`` over QQ, and return ``x_a - x_b``. For a
    connected graph this difference is gauge-invariant and equals the effective
    resistance.
    """

    from jacobian.math.graphs.electrical_networks._flint import solve_potentials

    potentials = solve_potentials(vertex_count, edges, terminal_a, terminal_b, fixed=0)
    return potentials[terminal_a] - potentials[terminal_b]


def node_potentials(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
    source: int,
    sink: int,
) -> list[Fraction]:
    """Solve the Dirichlet problem for unit current injection at source, sink.

    Inject one ampere at ``source`` and extract one ampere at ``sink``, with the
    sink gauge fixed to zero, returning exact rational node potentials.
    """

    from jacobian.math.graphs.electrical_networks._flint import solve_potentials

    return list(solve_potentials(vertex_count, edges, source, sink, fixed=sink))


def _claim_edges(
    claim: EffectiveResistanceResult | NodePotentialResult | LaplacianResult,
) -> tuple[tuple[int, int, Fraction], ...]:
    return tuple(
        (edge.source, edge.target, edge.conductance.as_fraction())
        for edge in claim.network.edges
    )


def verify_effective_resistance(claim: EffectiveResistanceResult) -> bool:
    """Check an effective-resistance value against its retained network."""
    network = claim.network
    if not 0 <= claim.terminal_a < network.vertex_count:
        return False
    if not 0 <= claim.terminal_b < network.vertex_count:
        return False
    if claim.terminal_a == claim.terminal_b:
        return False
    try:
        expected = effective_resistance(
            network.vertex_count,
            _claim_edges(claim),
            claim.terminal_a,
            claim.terminal_b,
        )
    except (ArithmeticError, ValueError, IndexError):
        return False
    return claim.effective_resistance.as_fraction() == expected


def verify_node_potentials(claim: NodePotentialResult) -> bool:
    """Check the Dirichlet equations for a retained network and potential vector."""
    network = claim.network
    if not 0 <= claim.source < network.vertex_count:
        return False
    if not 0 <= claim.sink < network.vertex_count or claim.source == claim.sink:
        return False
    if len(claim.potentials) != network.vertex_count:
        return False
    nodes = [value.node for value in claim.potentials]
    if set(nodes) != set(range(network.vertex_count)):
        return False
    values = {value.node: value.potential.as_fraction() for value in claim.potentials}
    if values[claim.sink] != 0:
        return False
    try:
        matrix = laplacian_matrix(network.vertex_count, _claim_edges(claim))
    except (ArithmeticError, ValueError, IndexError):
        return False
    for row in range(network.vertex_count):
        lhs = sum(matrix[row][col] * values[col] for col in range(network.vertex_count))
        expected = (1 if row == claim.source else 0) - (1 if row == claim.sink else 0)
        if lhs != expected:
            return False
    return True


def verify_laplacian(claim: LaplacianResult) -> bool:
    """Check every retained Laplacian entry against its source network."""
    network = claim.network
    if (
        claim.matrix.row_count != network.vertex_count
        or claim.matrix.column_count != network.vertex_count
    ):
        return False
    try:
        matrix = laplacian_matrix(network.vertex_count, _claim_edges(claim))
    except (ArithmeticError, ValueError, IndexError):
        return False
    return all(
        value.as_fraction() == matrix[row][col]
        for row, values in enumerate(claim.matrix.entries)
        for col, value in enumerate(values)
    )
