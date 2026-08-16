"""Markov chain operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.markov_chain import (
    ErgodicDecisionResult,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.markov_chain.operations import (
    compute_ergodic_decision,
    compute_stationary_distribution,
)
from jacobian.math_tools import MathTool


def mc_operation[RequestT: ContractModel, ResultT: ContractModel](
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
        "probability.markov_chain.stationary_distribution.compute",
        "Compute the stationary distribution of a Markov chain",
        "Compute the exact stationary distribution of a finite Markov chain using SymPy eigenvector computation.",
        TransitionMatrixRequest,
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
