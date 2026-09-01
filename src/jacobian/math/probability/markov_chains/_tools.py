"""Markov chain operation declarations."""

from typing import Any

from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.markov_chains._models import (
    CommunicatingClassesResult,
    ErgodicDecisionResult,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionRequest,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.probability.markov_chains.operations import (
    communicating_classes,
    ergodic_decision,
    mixing_time_result,
    stationary_distribution_result,
)
from jacobian.math.probability.markov_chains.values import as_transition_matrix


def compute_mixing_time(request: MixingTimeRequest) -> MixingTimeResult:
    """Unpack a wire request for the native mixing-time operation."""

    return mixing_time_result(
        as_transition_matrix(request.matrix),
        request.epsilon.as_fraction(),
        request.max_steps,
    )


def compute_stationary_distribution(
    request: StationaryDistributionRequest,
) -> StationaryDistributionResult:
    """Unpack a wire request for the native stationary-family operation."""

    return stationary_distribution_result(as_transition_matrix(request.matrix))


def compute_ergodic_decision(request: TransitionMatrixRequest) -> ErgodicDecisionResult:
    """Unpack a wire request for the native ergodicity operation."""

    return ergodic_decision(as_transition_matrix(request.matrix))


def compute_communicating_classes(
    request: TransitionMatrixRequest,
) -> CommunicatingClassesResult:
    """Unpack a wire request for the native class-decomposition operation."""

    return communicating_classes(as_transition_matrix(request.matrix))


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="probability.markov_chain.mixing_time.compute",
        title="Compute an exact bounded Markov-chain mixing time",
        description="Search exact rational matrix powers for the first step whose worst-case total-variation distance is at most epsilon; the chain must have at most eight states.",
        request_type=MixingTimeRequest,
        result_type=MixingTimeResult,
        run=compute_mixing_time,
        tags=(
            "probability",
            "markov-chain",
            "mixing-time",
            "total-variation",
            "exact",
            "bounded",
        ),
        examples=(
            OperationExample(
                name="two_state_chain",
                description="Compute the 1/100 mixing time of a two-state ergodic chain; max_steps bounds the exact search.",
                input={
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ],
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 8,
                },
            ),
        ),
    ),
    MathTool(
        operation_id="probability.markov_chain.stationary_distribution.compute",
        title="Compute the stationary-distribution family of a Markov chain",
        description="Compute the canonical extreme stationary distribution on every closed "
        "communicating class; their convex hull is the complete stationary family.",
        request_type=StationaryDistributionRequest,
        result_type=StationaryDistributionResult,
        run=compute_stationary_distribution,
        tags=("probability", "markov-chain", "stationary-distribution", "exact"),
        examples=(
            OperationExample(
                name="two_state_chain",
                description="Stationary distribution of a two-state rational Markov chain.",
                input={
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="probability.markov_chain.ergodic.decide",
        title="Decide whether a Markov chain is ergodic",
        description="Decide whether a finite Markov chain is ergodic (irreducible and aperiodic); aperiodicity is checked in every communicating class.",
        request_type=TransitionMatrixRequest,
        result_type=ErgodicDecisionResult,
        run=compute_ergodic_decision,
        tags=("probability", "markov-chain", "ergodic", "exact"),
        examples=(
            OperationExample(
                name="aperiodic_three_state_chain",
                description="An irreducible aperiodic chain with zeros in its square.",
                input={
                    "matrix": [
                        [
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                            {"num": "0", "den": "1"},
                        ],
                        [
                            {"num": "0", "den": "1"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "1"},
                        ],
                        [
                            {"num": "1", "den": "2"},
                            {"num": "0", "den": "1"},
                            {"num": "1", "den": "2"},
                        ],
                    ]
                },
            ),
        ),
    ),
    MathTool(
        operation_id="probability.markov_chain.communicating_classes.compute",
        title="Compute the communicating-class decomposition",
        description="Decompose a bounded exact finite Markov chain into communicating classes "
        "(strongly connected components) of its transition support graph, "
        "classifying each as transient or closed (recurrent).",
        request_type=TransitionMatrixRequest,
        result_type=CommunicatingClassesResult,
        run=compute_communicating_classes,
        tags=("markov-chain", "communicating-classes", "exact"),
        examples=(
            OperationExample(
                name="two_class_chain",
                description="A two-state chain where state 0 is transient and state 1 is absorbing.",
                input={
                    "matrix": [
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                        [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    ],
                },
            ),
        ),
    ),
)

__all__ = ["TOOLS"]
