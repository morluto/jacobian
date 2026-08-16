"""Markov chain operations backed by SymPy."""

from __future__ import annotations

__all__ = ["ergodic_properties", "mixing_time", "stationary_distribution"]


def mixing_time(matrix, epsilon, max_steps):  # type: ignore[no-untyped-def]
    """Exact mixing time of a finite Markov chain.

    Returns the smallest ``t >= 0`` such that ``max_x ||P^t(x,·) - π||_TV <= ε``,
    where ``π`` is the stationary distribution and ``||·||_TV`` is the total
    variation distance.  All arithmetic is exact (SymPy rationals).
    """
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"]) for j in range(n)]
            for i in range(n)
        ]
    )

    # Stationary distribution: left eigenvector of P for eigenvalue 1.
    pi = None
    for eigenval, _mult, vects in p.T.eigenvects():
        if eigenval == 1 and vects:
            vec = vects[0]
            total = sum(vec)
            pi = sympy.Matrix([v / total for v in vec])
            break
    if pi is None:
        raise ValueError("matrix has no stationary distribution (no eigenvalue 1)")

    eps = sympy.Rational(epsilon["num"], epsilon["den"])

    power = sympy.eye(n)
    for t in range(max_steps + 1):
        max_tv = sympy.S.Zero
        for x in range(n):
            tv = sympy.S.Zero
            for j in range(n):
                tv += sympy.Abs(power[x, j] - pi[j])
            tv = tv / 2
            if tv > max_tv:
                max_tv = tv
        if max_tv <= eps:
            return t
        power = power * p
    raise ValueError(
        f"mixing time exceeds the declared max_steps budget of {max_steps}"
    )


def stationary_distribution(matrix):  # type: ignore[no-untyped-def]
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"]) for j in range(n)]
            for i in range(n)
        ]
    )
    # Find eigenvector for eigenvalue 1
    eigenvects = p.T.eigenvects()
    for eigenval, _mult, vects in eigenvects:
        if eigenval == 1 and len(vects) > 0:
            vect = vects[0]
            total = sum(vect)
            normalized = [v / total for v in vect]
            return normalized
    return []


def ergodic_properties(matrix):  # type: ignore[no-untyped-def]
    import networkx as nx

    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(len(matrix)))
    graph.add_edges_from(
        (source, target)
        for source, row in enumerate(matrix)
        for target, value in enumerate(row)
        if value["num"] != "0"
    )
    irreducible = nx.is_strongly_connected(graph)
    aperiodic = all(
        nx.is_aperiodic(graph.subgraph(component))
        for component in nx.strongly_connected_components(graph)
    )
    return irreducible, aperiodic
