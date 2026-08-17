"""Exact electrical-network kernels backed by SymPy and NetworkX."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

__all__ = ["effective_resistance", "laplacian_matrix", "node_potentials"]


def _laplacian(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
) -> Any:
    """Build the conductance-weighted Laplacian as a SymPy Matrix of Rationals."""
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
    """Compute exact effective resistance via the Moore-Penrose pseudoinverse of L.

    Effective resistance R(a, b) = (e_a - e_b)^T L^+ (e_a - e_b),
    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.
    """
    from sympy import Matrix, Rational

    lap = _laplacian(vertex_count, edges)
    difference = Matrix.zeros(vertex_count, 1)
    difference[terminal_a, 0] = Rational(1)
    difference[terminal_b, 0] = Rational(-1)

    # Pinv via the reduced cofactor formula. SymPy's pinv uses SVD which is
    # numerical. Instead, we use the formula: L^+ = (L + J/n)^{-1} - J/n,
    # where J is the all-ones matrix, valid for connected graphs.
    # But this requires connectivity. The pseudoinverse approach via rank-deficient
    # solve is more direct: solve L x = (e_a - e_b) in the least-norm sense.
    #
    # For a connected graph, L has rank n-1, and the system L x = b has a solution
    # iff b sums to zero (which (e_a - e_b) does). The minimum-norm solution is
    # obtained by fixing a gauge (set one variable to 0, solve the reduced system).
    # Then R(a,b) = x_a - x_b for any such solution.
    #
    # We solve L_reduced x = b where we fix one node's potential to 0.
    fixed = 0
    if terminal_a == fixed:
        fixed = 1
    free = [i for i in range(vertex_count) if i != fixed]
    reduced = lap[free, free]
    rhs = Matrix.zeros(len(free), 1)
    b_full = Matrix.zeros(vertex_count, 1)
    b_full[terminal_a, 0] = Rational(1)
    b_full[terminal_b, 0] = Rational(-1)
    for idx, node in enumerate(free):
        rhs[idx, 0] = b_full[node, 0]
        for j in range(vertex_count):
            if j != fixed:
                rhs[idx, 0] -= lap[node, j] * Rational(0)

    sol = reduced.solve(rhs)
    potentials = [Rational(0)] * vertex_count
    for idx, _node in enumerate(free):
        potentials[free[idx]] = sol[idx, 0]
    r_val = potentials[terminal_a] - potentials[terminal_b]
    return Fraction(int(r_val.p), int(r_val.q))


def node_potentials(
    vertex_count: int,
    edges: tuple[tuple[int, int, Fraction], ...],
    source: int,
    sink: int,
) -> list[Fraction]:
    """Solve the Dirichlet problem for unit current injection at source, sink.

    Injects 1 ampere at source and extracts 1 ampere at sink. Returns node
    potentials with the gauge fixed so that the sink node has potential 0.
    """
    from sympy import Matrix, Rational

    lap = _laplacian(vertex_count, edges)
    b = Matrix.zeros(vertex_count, 1)
    b[source, 0] = Rational(1)
    b[sink, 0] = Rational(-1)

    # Fix sink potential to 0 (gauge), solve reduced system.
    free = [i for i in range(vertex_count) if i != sink]
    reduced = lap[free, free]
    rhs = Matrix.zeros(len(free), 1)
    for idx, node in enumerate(free):
        rhs[idx, 0] = b[node, 0]

    sol = reduced.solve(rhs)
    potentials = [Rational(0)] * vertex_count
    for idx, _node in enumerate(free):
        potentials[free[idx]] = sol[idx, 0]
    potentials[sink] = Rational(0)
    return [Fraction(int(v.p), int(v.q)) for v in potentials]
