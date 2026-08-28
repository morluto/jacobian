"""Multiple testing operation declarations."""

from collections.abc import Callable
from typing import Any

from jacobian._models import StrictModel
from jacobian.catalog._examples import example
from jacobian.catalog.models import MathTool, OperationExample
from jacobian.math.probability.multiple_testing import operations as native
from jacobian.math.probability.multiple_testing._models import (
    BHStepUpRequest,
    BHStepUpResult,
    FDPRequest,
    FDPResult,
)


def compute_bh_step_up(request: BHStepUpRequest) -> BHStepUpResult:
    return native.bh_step_up(request.hypotheses, request.level)


def compute_fdp(request: FDPRequest) -> FDPResult:
    return native.false_discovery_proportion(
        request.rejected_ids, request.true_null_ids
    )


def _op[
    RequestT: StrictModel,
    ResultT: StrictModel,
](
    operation_id: str,
    title: str,
    description: str,
    request_model: type[RequestT],
    result_model: type[ResultT],
    operation: Callable[[RequestT], ResultT],
    *tags: str,
    examples: tuple[OperationExample, ...] = (),
) -> MathTool[RequestT, ResultT]:
    return MathTool(
        operation_id=operation_id,
        title=title,
        description=description,
        request_type=request_model,
        result_type=result_model,
        run=operation,
        tags=tags,
        examples=examples,
    )


TOOLS: tuple[MathTool[Any, Any], ...] = (
    _op(
        "probability.fdp.compute",
        "Compute the false discovery proportion",
        "Return the exact count of false discoveries and the false discovery "
        "proportion among the supplied rejected hypotheses.",
        FDPRequest,
        FDPResult,
        compute_fdp,
        "probability",
        "multiple-testing",
        "false-discovery-proportion",
        "exact",
        examples=(
            example(
                "two_rejections_one_null",
                "Among h1 and h2 rejected, only h2 is a true null.",
                {"rejected_ids": ["h1", "h2"], "true_null_ids": ["h2", "h3"]},
            ),
        ),
    ),
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
)


__all__ = ["TOOLS"]
