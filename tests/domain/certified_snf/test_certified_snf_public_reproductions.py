from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"


def _smith_cases() -> list[dict[str, Any]]:
    cases = []
    for slug in ("smith-rank-deficient", "smith-rectangular"):
        task = TASK_ROOT / slug
        expected = json.loads((task / "tests" / "expected.json").read_text())
        cases.append(
            {
                "matrix": json.loads((task / "environment" / "input.json").read_text())[
                    "matrix"
                ],
                "expected_rank": expected["expected_rank"],
                "expected_invariant_factors": expected["expected_invariant_factors"],
            }
        )
    return cases


def test_public_certified_smith_cases_reach_checker_bound_results(
    certified_snf_services,
) -> None:
    for case in _smith_cases():
        computed = certified_snf_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.compute",
                input={"matrix": case["matrix"]},
            )
        )
        result = certified_snf_services.core.store.get(
            computed.output["result_uri"]
        ).payload
        certificate = result["certificate"]
        assert certificate["rank"] == case["expected_rank"]
        assert certificate["invariant_factors"] == case["expected_invariant_factors"]

        verified = certified_snf_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.verify",
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.verification_record_uri is not None
