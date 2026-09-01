"""Multiple testing operation declarations."""

from typing import Any

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


TOOLS: tuple[MathTool[Any, Any], ...] = (
    MathTool(
        operation_id="probability.fdp.compute",
        title="Compute the false discovery proportion",
        description="Return the exact count of false discoveries and the false discovery "
        "proportion among the supplied rejected hypotheses.",
        request_type=FDPRequest,
        result_type=FDPResult,
        run=compute_fdp,
        tags=("probability", "multiple-testing", "false-discovery-proportion", "exact"),
        examples=(
            OperationExample(
                name="two_rejections_one_null",
                description="Among h1 and h2 rejected, only h2 is a true null.",
                input={"rejected_ids": ["h1", "h2"], "true_null_ids": ["h2", "h3"]},
            ),
        ),
    ),
    MathTool(
        operation_id="probability.bh_step_up.compute",
        title="Benjamini-Hochberg step-up procedure",
        description="Given labelled p-values and a level q, compute the BH critical "
        "index, cutoff threshold, and rejection set.",
        request_type=BHStepUpRequest,
        result_type=BHStepUpResult,
        run=compute_bh_step_up,
        tags=("probability", "multiple-testing", "benjamini-hochberg", "exact"),
        examples=(
            OperationExample(
                name="three_hypotheses_all_rejected",
                description="3 hypotheses, p-values 0.01, 0.04, 0.03, level 0.05.",
                input={
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
