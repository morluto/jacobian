"""Markov chain operations backed by SymPy."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.markov_chains._models import (
    MAX_MIXING_STEPS,
    CommunicatingClassesResult,
    ErgodicDecisionResult,
    ExtremeStationaryDistribution,
    MixingTimeResult,
    StationaryDistributionResult,
)
from jacobian.math.probability.markov_chains.values import (
    TransitionMatrix,
    TransitionMatrixAdmissionError,
    _decimal_digits,
    as_canonical_transition_matrix,
    require_stationary_distribution_admission,
    require_transition_matrix,
)

__all__ = [
    "MixingTimeSearchResult",
    "communicating_classes",
    "ergodic_decision",
    "ergodic_properties",
    "mixing_time",
    "mixing_time_result",
    "stationary_distribution",
    "stationary_distribution_extremes",
    "stationary_distribution_result",
]


@dataclass(frozen=True, slots=True)
class MixingTimeSearchResult:
    mixing_time: int | None
    steps_examined: int
    max_total_variation_distance: Fraction


def mixing_time(
    matrix: TransitionMatrix,
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
    matrix: TransitionMatrix,
) -> list[tuple[tuple[int, ...], tuple[Fraction, ...]]]:
    """Return one normalized stationary vector for every closed class."""

    import networkx as nx

    n = len(matrix)
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(
        (source, target)
        for source, row in enumerate(matrix)
        for target, value in enumerate(row)
        if value != 0
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
    from jacobian.math.probability.markov_chains._flint import solve_stationary_class

    for closed_class in closed_classes:
        local = solve_stationary_class(matrix, closed_class)
        distribution = [Fraction(0)] * n
        for index, state in enumerate(closed_class):
            distribution[state] = local[index]
        extremes.append(
            (
                closed_class,
                tuple(distribution),
            )
        )
    return extremes


def stationary_distribution_extremes(
    matrix: TransitionMatrix,
) -> list[tuple[tuple[int, ...], tuple[Fraction, ...]]]:
    """Return one normalized stationary vector for every closed class."""

    _admit_stationary(matrix)
    return _stationary_distribution_extremes(matrix)


def stationary_distribution(
    matrix: TransitionMatrix,
) -> tuple[Fraction, ...]:
    """Return the unique stationary distribution, rejecting non-unique chains."""

    _admit_stationary(matrix)
    extremes = _stationary_distribution_extremes(matrix)
    if len(extremes) != 1:
        raise ValueError(
            "the Markov chain does not have a unique stationary distribution"
        )
    return extremes[0][1]


def _ergodic_properties(matrix: TransitionMatrix) -> tuple[bool, bool]:
    import networkx as nx

    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(len(matrix)))
    graph.add_edges_from(
        (source, target)
        for source, row in enumerate(matrix)
        for target, value in enumerate(row)
        if value != 0
    )
    irreducible = nx.is_strongly_connected(graph)
    aperiodic = all(
        nx.is_aperiodic(graph.subgraph(component))
        for component in nx.strongly_connected_components(graph)
    )
    return irreducible, aperiodic


def ergodic_properties(matrix: TransitionMatrix) -> tuple[bool, bool]:
    """Return whether a finite exact transition matrix is irreducible and aperiodic."""

    _admit_transition_matrix(matrix)
    return _ergodic_properties(matrix)


_MAX_MIXING_COMPONENT_DIGITS = 32
_MIXING_RESULT_DIGIT_RESERVE = 1_024


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"markov_chain.{code}", message=message
    )


def _admit_transition_matrix(matrix: TransitionMatrix) -> None:
    try:
        require_transition_matrix(matrix)
    except TransitionMatrixAdmissionError as exc:
        _reject(exc.location, exc.reason, str(exc))


def _admit_stationary(matrix: TransitionMatrix) -> None:
    try:
        require_stationary_distribution_admission(matrix)
    except TransitionMatrixAdmissionError as exc:
        _reject(exc.location, exc.reason, str(exc))


def _admit_mixing(
    matrix: TransitionMatrix,
    epsilon: Fraction,
    max_steps: int,
) -> None:
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_MIXING_STEPS:
        _reject(
            ("max_steps",),
            "mixing_steps_out_of_range",
            f"max_steps must be an integer between 1 and {MAX_MIXING_STEPS}",
        )
    if not 0 < epsilon <= 1:
        _reject(
            ("epsilon",),
            "mixing_epsilon_out_of_range",
            "epsilon must lie in (0, 1]",
        )
    values = (epsilon, *(item for row in matrix for item in row))
    if any(
        max(_decimal_digits(value.numerator), _decimal_digits(value.denominator))
        > _MAX_MIXING_COMPONENT_DIGITS
        for value in values
    ):
        _reject(
            ("matrix",),
            "mixing_component_digits_exceed_limit",
            "mixing-time rational components support at most "
            f"{_MAX_MIXING_COMPONENT_DIGITS} digits",
        )
    matrix_digits = max(
        max(_decimal_digits(value.numerator), _decimal_digits(value.denominator))
        for row in matrix
        for value in row
    )
    state_count = len(matrix)
    height = matrix_digits * (state_count**3 + max_steps * state_count**2)
    if height > MAX_CANONICAL_RATIONAL_DIGITS - _MIXING_RESULT_DIGIT_RESERVE:
        _reject(
            ("matrix",),
            "mixing_result_height_exceeds_bound",
            "mixing-time matrix height and max_steps can exceed the exact rational result bound",
        )


def _derive_communicating_classes(
    matrix: TransitionMatrix,
) -> tuple[tuple[tuple[tuple[int, ...], bool], ...], tuple[int, ...]]:
    """Derive the canonical SCC partition in bounded quadratic graph work."""

    import networkx as nx

    dimension = len(matrix)
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(dimension))
    graph.add_edges_from(
        (source, target)
        for source in range(dimension)
        for target in range(dimension)
        if matrix[source][target] > 0
    )
    sccs = list(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, sccs)
    classes: list[tuple[tuple[int, ...], bool]] = []
    state_class = [0] * dimension
    for class_index, scc_node in enumerate(nx.topological_sort(condensation)):
        states_set = sccs[scc_node]
        states = tuple(sorted(states_set))
        is_closed = not any(
            target not in states_set and matrix[source][target] > 0
            for source in states
            for target in range(dimension)
        )
        classes.append((states, is_closed))
        for state in states:
            state_class[state] = class_index
    return tuple(classes), tuple(state_class)


def mixing_time_result(
    matrix: TransitionMatrix,
    epsilon: Fraction,
    max_steps: int,
) -> MixingTimeResult:
    """Compute a bounded exact mixing result for a canonical matrix value."""

    _admit_transition_matrix(matrix)
    _admit_mixing(matrix, epsilon, max_steps)
    irreducible, aperiodic = _ergodic_properties(matrix)
    if not (irreducible and aperiodic):
        return MixingTimeResult(
            status="NOT_ERGODIC",
            epsilon=CanonicalRational.from_fraction(epsilon),
            max_steps=max_steps,
            steps_examined=0,
        )
    extremes = _stationary_distribution_extremes(matrix)
    stationary = extremes[0][1]
    outcome = mixing_time(matrix, stationary, epsilon, max_steps)
    distance = CanonicalRational.from_integer_ratio(
        outcome.max_total_variation_distance.numerator,
        outcome.max_total_variation_distance.denominator,
    )
    return MixingTimeResult(
        status="FOUND" if outcome.mixing_time is not None else "BOUND_EXCEEDED",
        epsilon=CanonicalRational.from_fraction(epsilon),
        max_steps=max_steps,
        steps_examined=outcome.steps_examined,
        mixing_time=outcome.mixing_time,
        max_total_variation_distance=distance,
    )


def stationary_distribution_result(
    matrix: TransitionMatrix,
) -> StationaryDistributionResult:
    """Compute the complete stationary family for a canonical matrix value."""

    _admit_stationary(matrix)
    extremes = _stationary_distribution_extremes(matrix)
    return StationaryDistributionResult._from_kernel(
        transition_matrix=as_canonical_transition_matrix(matrix),
        extreme_distributions=tuple(
            ExtremeStationaryDistribution(
                closed_class=closed_class,
                distribution=tuple(
                    CanonicalRational.from_integer_ratio(
                        value.numerator, value.denominator
                    )
                    for value in distribution
                ),
            )
            for closed_class, distribution in extremes
        ),
        unique=len(extremes) == 1,
    )


def ergodic_decision(matrix: TransitionMatrix) -> ErgodicDecisionResult:
    """Decide ergodicity for a canonical exact transition matrix value."""

    _admit_transition_matrix(matrix)
    irreducible, aperiodic = _ergodic_properties(matrix)
    return ErgodicDecisionResult(
        is_ergodic=irreducible and aperiodic,
        is_irreducible=irreducible,
        is_aperiodic=aperiodic,
    )


def communicating_classes(matrix: TransitionMatrix) -> CommunicatingClassesResult:
    """Decompose a canonical Markov matrix into communicating classes."""

    _admit_transition_matrix(matrix)
    classes, state_class = _derive_communicating_classes(matrix)
    return CommunicatingClassesResult._from_kernel(
        transition_matrix=as_canonical_transition_matrix(matrix),
        classes=classes,
        state_class=state_class,
    )
