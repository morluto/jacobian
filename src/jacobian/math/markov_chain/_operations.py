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


def compute_mixing_time(request: MixingTimeRequest) -> MixingTimeResult:
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    irreducible, aperiodic = ergodic_properties(request)
    if not (irreducible and aperiodic):
        return MixingTimeResult(
            status="NOT_ERGODIC",
            epsilon=request.epsilon,
            max_steps=request.max_steps,
            steps_examined=0,
        )
    extremes = _stationary_distribution_extremes(request)
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
    extremes = _stationary_distribution_extremes(request)
    return StationaryDistributionResult(
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
    irreducible, aperiodic = ergodic_properties(request)
    return ErgodicDecisionResult(
        is_ergodic=irreducible and aperiodic,
        is_irreducible=irreducible,
        is_aperiodic=aperiodic,
    )


def compute_communicating_classes(
    request: TransitionMatrixRequest,
) -> CommunicatingClassesResult:
    """Decompose a Markov chain into communicating classes via SCC analysis."""

    import networkx as nx

    matrix = request.matrix
    dimension = len(matrix)

    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_nodes_from(range(dimension))
    for i in range(dimension):
        for j in range(dimension):
            if matrix[i][j].as_fraction() > 0:
                graph.add_edge(i, j)

    sccs = list(nx.strongly_connected_components(graph))
    condensation = nx.condensation(graph, sccs)

    # Get topological order of SCC nodes
    scc_list = list(nx.topological_sort(condensation))

    # Reverse for recurrent-first order (closed classes last)
    # Actually, let's order by: transient classes first, then recurrent classes
    # A class is closed (recurrent) if it has no outgoing edges to other classes
    classes_info: list[tuple[tuple[int, ...], bool]] = []
    state_class = [0] * dimension

    for scc_idx, scc_node in enumerate(scc_list):
        scc = sccs[scc_node] if isinstance(scc_node, int) else scc_node
        states = sorted(scc)
        is_closed = True
        for state in states:
            for j in range(dimension):
                if j not in scc and matrix[state][j].as_fraction() > 0:
                    is_closed = False
                    break
            if not is_closed:
                break
        classes_info.append((tuple(states), is_closed))
        for state in states:
            state_class[state] = scc_idx

    return CommunicatingClassesResult(
        transition_matrix=request.matrix,
        classes=tuple(classes_info),
        state_class=tuple(state_class),
    )
