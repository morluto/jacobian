from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jacobian.contracts.operations import (
    OperationRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"


def _suite() -> list[dict[str, Any]]:
    cases = []
    for slug in (
        "reliability-series-path",
        "reliability-single-edge",
        "reliability-triangle-fair",
    ):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        request.pop("task_id", None)
        expected = json.loads((task / "tests" / "expected.json").read_text())
        cases.append(
            {
                "request": request,
                "expected_probability": expected["expected_probability"],
                "expected_states": expected["expected_states"],
            }
        )
    return cases


def test_public_small_graph_reliability_reaches_checker_bound_results(
    probability_services,
) -> None:
    for case in _suite():
        computed = probability_services.core.operations.invoke(
            OperationRequest(
                operation_id=(
                    "probability.graph_reliability.connection_probability.compute"
                ),
                input=case["request"],
            )
        )
        assert computed.execution.status is ExecutionStatus.COMPLETED
        assert (
            computed.output["result"]["connection_probability"]
            == (case["expected_probability"])
        )
        assert computed.output["result"]["visited_states"] == case["expected_states"]

        verified = probability_services.core.operations.invoke(
            OperationRequest(
                operation_id=(
                    "probability.graph_reliability.connection_probability.verify"
                ),
                input={
                    "input": case["request"],
                    "candidate": computed.output["result"],
                },
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.verification_record_uri is not None
