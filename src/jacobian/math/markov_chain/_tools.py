"""Markov chain operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.markov_chain._models import (
    CommunicatingClassesResult,
    ErgodicDecisionResult,
    MixingTimeRequest,
    MixingTimeResult,
    StationaryDistributionRequest,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.math.markov_chain._operations import (
    compute_communicating_classes,
    compute_ergodic_decision,
    compute_mixing_time,
    compute_stationary_distribution,
)


def mc_operation[RequestT: StrictModel, ResultT: StrictModel](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
    version: str = "1",
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        version=version,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


MARKOV_CHAIN_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    mc_operation(
        "probability.markov_chain.mixing_time.compute",
        "Compute an exact bounded Markov-chain mixing time",
        "Search exact rational matrix powers for the first step whose worst-case total-variation distance is at most epsilon; the chain must have at most eight states.",
        MixingTimeRequest,
        MixingTimeResult,
        compute_mixing_time,
        "probability",
        "markov-chain",
        "mixing-time",
        "total-variation",
        "exact",
        "bounded",
        examples=(
            example(
                "two_state_chain",
                "Compute the 1/100 mixing time of a two-state ergodic chain; max_steps bounds the exact search.",
                {
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ],
                    "epsilon": {"num": "1", "den": "100"},
                    "max_steps": 8,
                },
            ),
        ),
        version="2",
    ),
    mc_operation(
        "probability.markov_chain.stationary_distribution.compute",
        "Compute the stationary-distribution family of a Markov chain",
        "Compute the canonical extreme stationary distribution on every closed "
        "communicating class; their convex hull is the complete stationary family.",
        StationaryDistributionRequest,
        StationaryDistributionResult,
        compute_stationary_distribution,
        "probability",
        "markov-chain",
        "stationary-distribution",
        "exact",
        examples=(
            example(
                "two_state_chain",
                "Stationary distribution of a two-state rational Markov chain.",
                {
                    "matrix": [
                        [{"num": "1", "den": "2"}, {"num": "1", "den": "2"}],
                        [{"num": "1", "den": "4"}, {"num": "3", "den": "4"}],
                    ]
                },
            ),
        ),
        version="2",
    ),
    mc_operation(
        "probability.markov_chain.ergodic.decide",
        "Decide whether a Markov chain is ergodic",
        "Decide whether a finite Markov chain is ergodic (irreducible and aperiodic); aperiodicity is checked in every communicating class.",
        TransitionMatrixRequest,
        ErgodicDecisionResult,
        compute_ergodic_decision,
        "probability",
        "markov-chain",
        "ergodic",
        "exact",
        examples=(
            example(
                "aperiodic_three_state_chain",
                "An irreducible aperiodic chain with zeros in its square.",
                {
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
)

MARKOV_COMMUNICATING_CLASSES_OPERATION = mc_operation(
    "probability.markov_chain.communicating_classes.compute",
    "Compute the communicating-class decomposition",
    "Decompose a bounded exact finite Markov chain into communicating classes "
    "(strongly connected components) of its transition support graph, "
    "classifying each as transient or closed (recurrent).",
    TransitionMatrixRequest,
    CommunicatingClassesResult,
    compute_communicating_classes,
    "markov-chain",
    "communicating-classes",
    "exact",
    examples=(
        example(
            "two_class_chain",
            "A two-state chain where state 0 is transient and state 1 is absorbing.",
            {
                "matrix": [
                    [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                    [{"num": "0", "den": "1"}, {"num": "1", "den": "1"}],
                ],
            },
        ),
    ),
)

MARKOV_CHAIN_OPERATIONS = (
    *MARKOV_CHAIN_OPERATIONS,
    MARKOV_COMMUNICATING_CLASSES_OPERATION,
)

TOOLS = MARKOV_CHAIN_OPERATIONS

__all__ = ["TOOLS"]
