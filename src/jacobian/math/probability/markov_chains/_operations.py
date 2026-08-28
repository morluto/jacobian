"""Domain-owned Markov chain operations."""

from __future__ import annotations

from math import factorial

from jacobian._exact import MAX_CANONICAL_RATIONAL_DIGITS, CanonicalRational
from jacobian.catalog.models import OperationDomainValidationError
from jacobian.math.probability.markov_chains import (
    ergodic_properties,
    mixing_time,
)
from jacobian.math.probability.markov_chains._models import (
    CommunicatingClassesResult,
    ErgodicDecisionResult,
    ExtremeStationaryDistribution,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionRequest,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.probability.markov_chains.operations import (
    _stationary_distribution_extremes,
)

_MAX_MIXING_COMPONENT_DIGITS = 32
_MIXING_RESULT_DIGIT_RESERVE = 1_024


def _reject(location: tuple[str | int, ...], code: str, message: str) -> None:
    raise OperationDomainValidationError(
        location=location, code=f"markov_chain.{code}", message=message
    )


def _admit_stationary(request: StationaryDistributionRequest) -> None:
    dimension = len(request.matrix)
    row_bounds: list[int] = []
    for column in range(dimension - 1):
        entries = tuple(request.matrix[row][column] for row in range(dimension))
        denominator_digits = sum(len(value.den) for value in entries)
        row_bounds.append(
            max(
                max(len(value.num.lstrip("-")), len(value.den))
                + 1
                + denominator_digits
                - len(value.den)
                for value in entries
            )
        )
    determinant_digits = sum(row_bounds) + 1 + len(str(factorial(dimension)))
    if determinant_digits > MAX_CANONICAL_RATIONAL_DIGITS:
        _reject(
            ("matrix",),
            "stationary_height_exceeds_bound",
            "stationary distribution rational height exceeds the "
            f"{MAX_CANONICAL_RATIONAL_DIGITS}-digit result bound",
        )


def _admit_mixing(request: MixingTimeRequest) -> None:
    if not 0 < request.epsilon.as_fraction() <= 1:
        _reject(
            ("epsilon",),
            "mixing_epsilon_out_of_range",
            "epsilon must lie in (0, 1]",
        )
    values = (request.epsilon, *(item for row in request.matrix for item in row))
    if any(
        max(len(value.num.lstrip("-")), len(value.den)) > _MAX_MIXING_COMPONENT_DIGITS
        for value in values
    ):
        _reject(
            ("matrix",),
            "mixing_component_digits_exceed_limit",
            "mixing-time rational components support at most "
            f"{_MAX_MIXING_COMPONENT_DIGITS} digits",
        )
    matrix_digits = max(
        max(len(value.num.lstrip("-")), len(value.den))
        for row in request.matrix
        for value in row
    )
    state_count = len(request.matrix)
    height = matrix_digits * (state_count**3 + request.max_steps * state_count**2)
    if height > MAX_CANONICAL_RATIONAL_DIGITS - _MIXING_RESULT_DIGIT_RESERVE:
        _reject(
            ("matrix",),
            "mixing_result_height_exceeds_bound",
            "mixing-time matrix height and max_steps can exceed the exact rational result bound",
        )


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
    _admit_mixing(request)
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
    _admit_stationary(request)
    matrix = tuple(
        tuple(value.as_fraction() for value in row) for row in request.matrix
    )
    extremes = _stationary_distribution_extremes(matrix)
    return StationaryDistributionResult._from_kernel(
        transition_matrix=request.matrix,
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
