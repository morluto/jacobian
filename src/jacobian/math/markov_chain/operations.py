"""Markov chain operations backed by SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian.canonical import parse_canonical_integer
from jacobian.math.markov_chain._models import (
    StationaryDistributionRequest,
    TransitionMatrixRequest,
)

__all__ = [
    "MixingTimeSearchResult",
    "ergodic_properties",
    "mixing_time",
    "stationary_distribution",
    "stationary_distribution_extremes",
]


@dataclass(frozen=True, slots=True)
class MixingTimeSearchResult:
    mixing_time: int | None
    steps_examined: int
    max_total_variation_distance: Fraction


def mixing_time(
    matrix: tuple[tuple[Fraction, ...], ...],
    stationary: tuple[Fraction, ...],
    epsilon: Fraction,
    max_steps: int,
) -> MixingTimeSearchResult:
    """Return the first exact worst-case epsilon-mixing step within the bound."""
    import sympy

    transition = sympy.Matrix(
        [[sympy.Rational(v.numerator, v.denominator) for v in row] for row in matrix]
    )
    target = tuple(sympy.Rational(v.numerator, v.denominator) for v in stationary)
    threshold = sympy.Rational(epsilon.numerator, epsilon.denominator)
    power = sympy.eye(len(matrix))
    terminal = sympy.S.One
    for step in range(max_steps + 1):
        terminal = max(
            sum(
                abs(power[source, target_index] - target[target_index])
                for target_index in range(len(matrix))
            )
            / 2
            for source in range(len(matrix))
        )
        distance = Fraction(int(terminal.p), int(terminal.q))
        if terminal <= threshold:
            return MixingTimeSearchResult(step, step + 1, distance)
        if step < max_steps:
            power *= transition
    return MixingTimeSearchResult(
        None, max_steps + 1, Fraction(int(terminal.p), int(terminal.q))
    )


def _stationary_distribution_extremes(
    request: TransitionMatrixRequest,
) -> list[tuple[tuple[int, ...], tuple[Fraction, ...]]]:
    """Return one normalized stationary vector for every closed class."""

    import networkx as nx
    import sympy

    matrix = request.matrix
    n = len(matrix)
    p = sympy.Matrix(
        [
            [
                sympy.Rational(
                    parse_canonical_integer(matrix[i][j].num),
                    parse_canonical_integer(matrix[i][j].den),
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
    )
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (source, target)
        for source, row in enumerate(matrix)
        for target, value in enumerate(row)
        if value.as_fraction() != 0
    )
    closed_classes = sorted(
        (
            tuple(sorted(component))
            for component in nx.strongly_connected_components(graph)
            if not any(
                target not in component
                for source in component
                for target in graph.successors(source)
            )
        ),
        key=lambda component: component,
    )
    extremes: list[tuple[tuple[int, ...], tuple[Fraction, ...]]] = []
    for closed_class in closed_classes:
        submatrix = p.extract(closed_class, closed_class)
        equations = submatrix.T - sympy.eye(len(closed_class))
        equations[len(closed_class) - 1, :] = sympy.ones(1, len(closed_class))
        rhs = sympy.zeros(len(closed_class), 1)
        rhs[len(closed_class) - 1, 0] = 1
        local = equations.inv() * rhs
        distribution = [sympy.S.Zero] * n
        for index, state in enumerate(closed_class):
            distribution[state] = local[index]
        extremes.append(
            (
                closed_class,
                tuple(Fraction(int(value.p), int(value.q)) for value in distribution),
            )
        )
    return extremes


def stationary_distribution_extremes(
    request: StationaryDistributionRequest,
) -> list[tuple[tuple[int, ...], tuple[Fraction, ...]]]:
    """Return one normalized stationary vector for every closed class."""

    return _stationary_distribution_extremes(request)


def stationary_distribution(
    request: StationaryDistributionRequest,
) -> tuple[Fraction, ...]:
    """Return the unique stationary distribution, rejecting non-unique chains."""

    extremes = _stationary_distribution_extremes(request)
    if len(extremes) != 1:
        raise ValueError(
            "the Markov chain does not have a unique stationary distribution"
        )
    return extremes[0][1]


def ergodic_properties(request: TransitionMatrixRequest) -> tuple[bool, bool]:
    import networkx as nx

    graph: nx.DiGraph[int] = nx.DiGraph()
    matrix = request.matrix
    graph.add_nodes_from(range(len(matrix)))
    graph.add_edges_from(
        (source, target)
        for source, row in enumerate(matrix)
        for target, value in enumerate(row)
        if value.as_fraction() != 0
    )
    irreducible = nx.is_strongly_connected(graph)
    aperiodic = all(
        nx.is_aperiodic(graph.subgraph(component))
        for component in nx.strongly_connected_components(graph)
    )
    return irreducible, aperiodic
