from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityAssuranceLevel,
    CapabilityMode,
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPRODUCTIONS = (
    PROJECT_ROOT
    / "benchmarks"
    / "reproduction_cases"
    / "gaussian_polynomial_moment_public.json"
)


def _load_suite() -> dict[str, Any]:
    suite = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    assert suite["scored"] is False
    assert suite["purpose"].endswith("never evidence for an all-order identity")
    assert suite["held_out_evaluation"]["status"] == "READY_NOT_RUN"
    assert len(suite["attack_coverage"]) >= 3
    return suite


def test_public_gaussian_moment_reproductions_reach_checker_bound_results(
    authorized_complete_runtime,
) -> None:
    for case in _load_suite()["cases"]:
        computed = authorized_complete_runtime.core.capabilities.invoke(
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

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="probability.result.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )

        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.output["operation_id"] == (
            "probability.gaussian_polynomial.moment.compute"
        )
        assert verified.output["verification_record_uri"] in verified.artifact_uris
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
