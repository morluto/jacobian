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
    PROJECT_ROOT / "benchmarks" / "reproduction_cases" / "graph_reliability_public.json"
)


def _suite() -> dict[str, Any]:
    suite = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    assert suite["scored"] is False
    assert suite["held_out_evaluation"]["status"] == "READY_NOT_RUN"
    return suite


def test_public_small_graph_reliability_reaches_checker_bound_results(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["cases"]:
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id=(
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
        assert computed.output["result"]["completeness"] == "COMPLETE"
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="probability.result.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
