from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"


def _load_suite() -> list[dict[str, Any]]:
    cases = []
    for slug in (
        "gaussian-complex-cancellation",
        "gaussian-sixth-moment",
        "gaussian-two-sum-fourth-moment",
    ):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        request.pop("task_id", None)
        cases.append(
            {
                "request": request,
                "expected_moment": json.loads(
                    (task / "tests" / "expected.json").read_text()
                )["expected_moment"],
            }
        )
    return cases


def test_public_gaussian_moment_reproductions_reach_checker_bound_results(
    probability_services,
) -> None:
    for case in _load_suite():
        computed = probability_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="probability.gaussian_polynomial.moment.compute",
                input=case["request"],
            )
        )

        assert computed.execution.status is ExecutionStatus.COMPLETED
        assert computed.output["result"]["moment"] == case["expected_moment"]
        assert computed.output["result"]["completeness"] == (
            "COMPLETE_BOUNDED_EXPANSION"
        )
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = probability_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="probability.gaussian_polynomial.moment.verify",
                input={
                    "input": case["request"],
                    "candidate": computed.output["result"],
                },
            )
        )

        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.output["operation_id"] == (
            "probability.gaussian_polynomial.moment.compute"
        )
        assert verified.output["verification_record_uri"] in verified.artifact_uris
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
