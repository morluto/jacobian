"""Markov chain operation declarations."""

from jacobian.contracts.base import ContractModel
from jacobian.contracts.markov_chain import (
    ErgodicDecisionResult,
    StationaryDistributionResult,
    TransitionMatrixRequest,
)
from jacobian.domains.markov_chain.operations import (
    compute_ergodic_decision,
    compute_stationary_distribution,
)
from jacobian.math_tools import MathTool


def mc_operation[RequestT: ContractModel, ResultT: ContractModel](
    operation_id,
    title,
    description,
    request_model,
    result_model,
    operation,
    *tags,
    examples=(),
    version="1",
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


MARKOV_CHAIN_OPERATIONS = (
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
        examples=(),
    ),
    mc_operation(
        "probability.markov_chain.ergodic.decide",
        "Decide whether a Markov chain is ergodic",
        "Decide whether a finite Markov chain is ergodic (irreducible and aperiodic) using SymPy.",
        TransitionMatrixRequest,
        ErgodicDecisionResult,
        compute_ergodic_decision,
        "probability",
        "markov-chain",
        "ergodic",
        "exact",
        examples=(),
    ),
)
