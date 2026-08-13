from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from tests.boundary.providers.external_sat.external_sat_support import (
    open_verified_external_sat_services,
)
from tests.support.provider_external_sat import external_sat_toolchain_available

from jacobian.contracts.capabilities import (
    CapabilityRequest,
)
from jacobian.contracts.results import ExecutionStatus

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"

pytestmark = [
    pytest.mark.skipif(
        not external_sat_toolchain_available(),
        reason="the pinned CaDiCaL and DRAT-trim runtimes are unavailable",
    ),
]


def _load_cases() -> list[dict[str, Any]]:
    cases = []
    for slug in ("sat-bool-mus", "sat-pigeonhole", "sat-small"):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        expected = json.loads((task / "tests" / "expected.json").read_text())
        cases.append(
            {
                "variable_names": request["variables"],
                "clauses": request["clauses"],
                "expected_status": expected["expected_status"],
                "required_capabilities": [
                    (
                        "sat.model.find"
                        if expected["expected_status"] == "SATISFIABLE"
                        else "sat.unsat_proof.find"
                    ),
                    (
                        "sat.model.verify"
                        if expected["expected_status"] == "SATISFIABLE"
                        else "sat.unsat_proof.verify"
                    ),
                ],
            }
        )
    return cases


def test_sat_public_reproductions_reach_checker_bound_results(
    tmp_path: Path,
) -> None:
    with open_verified_external_sat_services(tmp_path / "state") as runtime:
        for case in _load_cases():
            cnf = runtime.core.sat.put_cnf(
                variable_names=tuple(case["variable_names"]),
                clauses=tuple(tuple(clause) for clause in case["clauses"]),
            )
            if case["expected_status"] == "SATISFIABLE":
                find_id = "sat.model.find"
                verify_id = "sat.model.verify"
                evidence_field = "assignment_uri"
            else:
                find_id = "sat.unsat_proof.find"
                verify_id = "sat.unsat_proof.verify"
                evidence_field = "proof_uri"

            found = runtime.core.capabilities.invoke(
                CapabilityRequest(
                    capability_id=find_id,
                    input={
                        "cnf_uri": cnf.artifact_uri,
                        "resource_budget": {"wall_seconds": 5},
                    },
                )
            )
            assert found.execution.status is ExecutionStatus.COMPLETED
            assert "conclusion" not in found.output
            evidence_uri = found.output[evidence_field]
            assert evidence_uri is not None
            if case["expected_status"] == "SATISFIABLE":
                assert found.output["assignment"] is not None

            verified = runtime.core.capabilities.invoke(
                CapabilityRequest(
                    capability_id=verify_id,
                    input={evidence_field: evidence_uri},
                )
            )
            assert verified.execution.status is ExecutionStatus.COMPLETED
            assert verified.output["conclusion"] == "TRUE"
            assert verified.output["cnf_uri"] == cnf.artifact_uri
            assert verified.output[evidence_field] == evidence_uri
            assert verified.verification_record_uri is not None
            assert case["required_capabilities"] == [find_id, verify_id]
