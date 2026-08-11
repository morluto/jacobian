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


def _suite() -> list[dict[str, Any]]:
    cases = []
    for slug in (
        "symmetry-colored-reflection",
        "symmetry-cycle-rotation",
        "symmetry-identity-subgroup",
    ):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        request.pop("task_id", None)
        cases.append(
            {
                "request": request,
                **json.loads((task / "tests" / "expected.json").read_text()),
            }
        )
    return cases


def test_public_declared_graph_symmetry_cases_reach_checker_bound_results(
    graph_symmetry_services,
) -> None:
    for case in _suite():
        computed = graph_symmetry_services.core.capabilities.invoke(
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

        verified = graph_symmetry_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="graph.symmetry.generator_orbits.verify",
                input={"input": case["request"], "candidate": result},
            )
        )
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
