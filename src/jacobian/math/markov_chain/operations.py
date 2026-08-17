"""Markov chain operations backed by SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True, slots=True)
class MixingTimeSearchResult:
    """The first satisfactory step, or the distance at the search bound."""

    mixing_time: int | None
    steps_examined: int
    max_total_variation_distance: Fraction


def mixing_time(
    matrix: tuple[tuple[Fraction, ...], ...],
    stationary_distribution: tuple[Fraction, ...],
    epsilon: Fraction,
    max_steps: int,
) -> MixingTimeSearchResult:
    """Search exact rational powers for the first epsilon-mixing step."""
    from sympy import Matrix, Rational, eye

    dimension = len(matrix)
    transition = Matrix(
        [
            [Rational(value.numerator, value.denominator) for value in row]
            for row in matrix
        ]
    )
    stationary = Matrix(
        [
            Rational(value.numerator, value.denominator)
            for value in stationary_distribution
        ]
    )
    epsilon_value = Rational(epsilon.numerator, epsilon.denominator)
    power = eye(dimension)
    terminal_distance = Rational(1)
    for step in range(max_steps + 1):
        distances = [
            sum(
                abs(power[state, target] - stationary[target])
                for target in range(dimension)
            )
            / 2
            for state in range(dimension)
        ]
        terminal_distance = max(distances)
        exact_distance = Fraction(
            int(terminal_distance.p),
            int(terminal_distance.q),
        )
        if terminal_distance <= epsilon_value:
            return MixingTimeSearchResult(
                mixing_time=step,
                steps_examined=step,
                max_total_variation_distance=exact_distance,
            )
        power *= transition
    return MixingTimeSearchResult(
        mixing_time=None,
        steps_examined=max_steps,
        max_total_variation_distance=Fraction(
            int(terminal_distance.p),
            int(terminal_distance.q),
        ),
    )


def stationary_distribution(matrix):  # type: ignore[no-untyped-def]
    """Rank-aware exact stationary distribution of a finite Markov chain.

    Solves ``(P**T - I) pi = 0`` with exact rational arithmetic.  When the
    eigenvalue-1 left eigenspace is one-dimensional (the unique stationary
    distribution, guaranteed for ergodic chains), the normalized generator is
    returned.  Otherwise the first eigenvalue-1 eigenvector is normalized and
    returned, preserving the pre-existing contract for non-ergodic chains.
    """
    import sympy

    n = len(matrix)
    p = sympy.Matrix(
        [
            [sympy.Rational(matrix[i][j]["num"], matrix[i][j]["den"]) for j in range(n)]
            for i in range(n)
        ]
    )
    kernel = (p.T - sympy.eye(n)).nullspace()
    if len(kernel) == 1:
        vector = kernel[0]
        total = sum(vector)
        return tuple(
            Fraction(int((value / total).p), int((value / total).q)) for value in vector
        )
    for eigenval, _mult, vects in p.T.eigenvects():
        if eigenval == 1 and len(vects) > 0:
            vector = vects[0]
            total = sum(vector)
            return tuple(
                Fraction(int((value / total).p), int((value / total).q))
                for value in vector
            )
    return ()


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


__all__ = [
    "MixingTimeSearchResult",
    "ergodic_properties",
    "mixing_time",
    "stationary_distribution",
]
