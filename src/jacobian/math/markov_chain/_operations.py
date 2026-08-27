"""Domain-owned Markov chain operations."""

from __future__ import annotations

from jacobian._exact import CanonicalRational
from jacobian.math.markov_chain import (
    ergodic_properties,
    mixing_time,
)
from jacobian.math.markov_chain._models import (
    CommunicatingClassesResult,
    ErgodicDecisionResult,
    ExtremeStationaryDistribution,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionRequest,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain.operations import _stationary_distribution_extremes


def _derive_communicating_classes(
    matrix: tuple[tuple[CanonicalRational, ...], ...],
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
        if matrix[source][target].as_fraction() > 0
    )
    sccs = list(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, sccs)
    classes: list[tuple[tuple[int, ...], bool]] = []
    state_class = [0] * dimension
    for class_index, scc_node in enumerate(nx.topological_sort(condensation)):
        states_set = sccs[scc_node]
        states = tuple(sorted(states_set))
        is_closed = not any(
            target not in states_set and matrix[source][target].as_fraction() > 0
            for source in states
            for target in range(dimension)
        )
        classes.append((states, is_closed))
        for state in states:
            state_class[state] = class_index
    return tuple(classes), tuple(state_class)


def compute_mixing_time(request: MixingTimeRequest) -> MixingTimeResult:
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    irreducible, aperiodic = ergodic_properties(matrix)
    if not (irreducible and aperiodic):
        return MixingTimeResult(
            status="NOT_ERGODIC",
            epsilon=request.epsilon,
            max_steps=request.max_steps,
            steps_examined=0,
        )
    extremes = _stationary_distribution_extremes(matrix)
    stationary = extremes[0][1]
    outcome = mixing_time(
        matrix, stationary, request.epsilon.as_fraction(), request.max_steps
    )
    distance = CanonicalRational.from_integer_ratio(
        outcome.max_total_variation_distance.numerator,
        outcome.max_total_variation_distance.denominator,
    )
    return MixingTimeResult(
        status="FOUND" if outcome.mixing_time is not None else "BOUND_EXCEEDED",
        epsilon=request.epsilon,
        max_steps=request.max_steps,
        steps_examined=outcome.steps_examined,
        mixing_time=outcome.mixing_time,
        max_total_variation_distance=distance,
    )


def compute_stationary_distribution(
    request: StationaryDistributionRequest,
) -> StationaryDistributionResult:
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    extremes = _stationary_distribution_extremes(matrix)
    return StationaryDistributionResult._from_kernel(
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


def compute_ergodic_decision(request: TransitionMatrixRequest) -> ErgodicDecisionResult:
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    irreducible, aperiodic = ergodic_properties(matrix)
    return ErgodicDecisionResult(
        is_ergodic=irreducible and aperiodic,
        is_irreducible=irreducible,
        is_aperiodic=aperiodic,
    )


def compute_communicating_classes(
    request: TransitionMatrixRequest,
) -> CommunicatingClassesResult:
    """Decompose a Markov chain into communicating classes via SCC analysis."""

    matrix = request.matrix
    classes, state_class = _derive_communicating_classes(matrix)
    return CommunicatingClassesResult._from_kernel(
        transition_matrix=request.matrix,
        classes=classes,
        state_class=state_class,
    )


def _verify_communicating_classes_result(result: CommunicatingClassesResult) -> bool:
    """Replay the SCC relation for an independently supplied bounded result."""

    request = TransitionMatrixRequest(matrix=result.transition_matrix)
    classes, state_class = _derive_communicating_classes(request.matrix)
    return result.classes == classes and result.state_class == state_class
