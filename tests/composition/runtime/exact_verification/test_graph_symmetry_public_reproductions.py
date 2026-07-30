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
    / "graph_symmetry_orbits_public.json"
)


def _suite() -> dict[str, Any]:
    suite = json.loads(REPRODUCTIONS.read_text(encoding="utf-8"))
    assert suite["scored"] is False
    assert suite["held_out_evaluation"]["status"] == "READY_NOT_RUN"
    return suite


def test_public_declared_graph_symmetry_cases_reach_checker_bound_results(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["cases"]:
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="graph.symmetry.generator_orbits.compute",
                input=case["request"],
            )
        )
        assert computed.execution.status is ExecutionStatus.COMPLETED
        result = computed.output["result"]
        assert [orbit["members"] for orbit in result["vertex_orbits"]] == (
            case["expected_vertex_orbits"]
        )
        assert [orbit["members"] for orbit in result["edge_orbits"]] == (
            case["expected_edge_orbits"]
        )
        assert result["orbit_completeness"] == "COMPLETE_FOR_DECLARED_GENERATORS"
        assert result["automorphism_group_completeness"] == (
            "FULL_AUTOMORPHISM_GROUP_NOT_CLAIMED"
        )
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="graph.symmetry.generator_orbits.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
