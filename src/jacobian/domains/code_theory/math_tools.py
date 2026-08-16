"""Code theory operation declarations."""

from jacobian.contracts.base import ContractModel
from jacobian.contracts.code_theory import (
    LinearCodeRequest,
    MinimumDistanceResult,
    WeightDistributionResult,
)
from jacobian.domains.code_theory.operations import (
    compute_min_distance,
    compute_weight_dist,
)
from jacobian.math_tools import MathTool


def ct_operation[RequestT: ContractModel, ResultT: ContractModel](
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


CODE_THEORY_OPERATIONS = (
    ct_operation(
        "code.minimum_distance.compute",
        "Compute the minimum distance of a linear code",
        "Compute the minimum Hamming distance of a linear code given by a generator matrix over a finite field.",
        LinearCodeRequest,
        MinimumDistanceResult,
        compute_min_distance,
        "code",
        "minimum-distance",
        "exact",
        examples=(),
    ),
    ct_operation(
        "code.weight_distribution.compute",
        "Compute the weight distribution of a linear code",
        "Compute the weight distribution of a linear code given by a generator matrix over a finite field.",
        LinearCodeRequest,
        WeightDistributionResult,
        compute_weight_dist,
        "code",
        "weight-distribution",
        "exact",
        examples=(),
    ),
)
