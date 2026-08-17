"""Multiple testing operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian.contracts.base import ContractModel
from jacobian.contracts.multiple_testing import (
    BHStepUpRequest,
    BHStepUpResult,
    FDPRequest,
    FDPResult,
)
from jacobian.contracts.operations import OperationExample
from jacobian.domains._examples import example
from jacobian.domains.multiple_testing.operations import (
    compute_bh_step_up,
    compute_fdp,
)
from jacobian.math_tools import MathTool


def _op[
    RequestT: ContractModel,
    ResultT: ContractModel,
](
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


MULTIPLE_TESTING_OPERATIONS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "probability.bh_step_up.compute",
        "Benjamini-Hochberg step-up procedure",
        "Given labelled p-values and a level q, compute the BH critical "
        "index, cutoff threshold, and rejection set.",
        BHStepUpRequest,
        BHStepUpResult,
        compute_bh_step_up,
        "probability",
        "multiple-testing",
        "benjamini-hochberg",
        "exact",
        examples=(
            example(
                "three_hypotheses_all_rejected",
                "3 hypotheses, p-values 0.01, 0.04, 0.03, level 0.05.",
                {
                    "hypotheses": [
                        {"hypothesis_id": "h1", "p_value": {"num": "1", "den": "100"}},
                        {"hypothesis_id": "h2", "p_value": {"num": "1", "den": "25"}},
                        {"hypothesis_id": "h3", "p_value": {"num": "3", "den": "100"}},
                    ],
                    "level": {"num": "1", "den": "20"},
                },
            ),
        ),
    ),
    _op(
        "probability.fdp.compute",
        "False discovery proportion",
        "Given rejected and true-null hypothesis IDs, compute the exact "
        "false discovery proportion FDP = |rejected nulls| / |rejected|.",
        FDPRequest,
        FDPResult,
        compute_fdp,
        "probability",
        "multiple-testing",
        "fdp",
        "exact",
        examples=(
            example(
                "simple_fdp",
                "2 rejections, 1 true null.",
                {"rejected_ids": ["h1", "h2"], "true_null_ids": ["h2", "h3"]},
            ),
        ),
    ),
)


__all__ = ["MULTIPLE_TESTING_OPERATIONS"]
