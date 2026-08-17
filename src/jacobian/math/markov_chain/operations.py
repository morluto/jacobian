"""Markov chain operations backed by SymPy."""

from __future__ import annotations

from jacobian.canonical import parse_canonical_integer

__all__ = ["ergodic_properties", "stationary_distribution"]


def stationary_distribution(matrix):  # type: ignore[no-untyped-def]
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [
                sympy.Rational(
                    parse_canonical_integer(matrix[i][j]["num"]),
                    parse_canonical_integer(matrix[i][j]["den"]),
                )
                for j in range(n)
            ]
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
