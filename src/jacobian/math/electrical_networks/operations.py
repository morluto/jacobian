"""Exact electrical-network kernels backed by SymPy."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

__all__ = ["effective_resistance", "laplacian_matrix", "node_potentials"]


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

    from sympy import Matrix, Rational

    lap = _laplacian(vertex_count, edges)
    fixed = 0
    free = [node for node in range(vertex_count) if node != fixed]
    reduced = lap[free, free]
    rhs = Matrix.zeros(len(free), 1)
    for idx, node in enumerate(free):
        if node == terminal_a:
            rhs[idx, 0] += Rational(1)
        if node == terminal_b:
            rhs[idx, 0] -= Rational(1)

    sol = reduced.solve(rhs)
    potentials = [Rational(0)] * vertex_count
    for idx, node in enumerate(free):
        potentials[node] = sol[idx, 0]

    value = potentials[terminal_a] - potentials[terminal_b]
    return Fraction(int(value.p), int(value.q))


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

    from sympy import Matrix, Rational

    lap = _laplacian(vertex_count, edges)
    free = [node for node in range(vertex_count) if node != sink]
    reduced = lap[free, free]
    rhs = Matrix.zeros(len(free), 1)
    for idx, node in enumerate(free):
        if node == source:
            rhs[idx, 0] += Rational(1)

    sol = reduced.solve(rhs)
    potentials = [Rational(0)] * vertex_count
    for idx, node in enumerate(free):
        potentials[node] = sol[idx, 0]
    potentials[sink] = Rational(0)
    return [Fraction(int(value.p), int(value.q)) for value in potentials]
