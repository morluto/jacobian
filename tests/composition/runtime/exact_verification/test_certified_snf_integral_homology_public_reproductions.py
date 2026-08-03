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
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"


def _suite() -> dict[str, list[dict[str, Any]]]:
    smith = []
    for slug in ("smith-rank-deficient", "smith-rectangular"):
        task = TASK_ROOT / slug
        expected = json.loads((task / "tests" / "expected.json").read_text())
        smith.append(
            {
                "matrix": json.loads((task / "environment" / "input.json").read_text())[
                    "matrix"
                ],
                "expected_rank": expected["expected_rank"],
                "expected_invariant_factors": expected["expected_invariant_factors"],
            }
        )
    homology = []
    for slug in ("integral-circle", "integral-projective-plane", "reduced-point"):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        expected = json.loads((task / "tests" / "expected.json").read_text())
        homology.append(
            {
                "presentation": request["presentation"],
                "convention": request["convention"],
                "expected_free_ranks": expected["expected_free_ranks"],
                "expected_torsion": expected["expected_torsion"],
            }
        )
    return {"smith_cases": smith, "homology_cases": homology}


def _result_payload(runtime, computed) -> dict[str, Any]:
    return runtime.core.store.get(computed.output["result_uri"]).payload


def test_public_certified_smith_cases_reach_checker_bound_results(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["smith_cases"]:
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.compute",
                input={"matrix": case["matrix"]},
            )
        )
        result = _result_payload(authorized_complete_runtime, computed)
        certificate = result["certificate"]
        assert certificate["rank"] == case["expected_rank"]
        assert certificate["invariant_factors"] == case["expected_invariant_factors"]
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="matrix.normal_form.smith.certified.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED


def test_public_integral_homology_cases_bind_generators_and_torsion(
    authorized_complete_runtime,
) -> None:
    for case in _suite()["homology_cases"]:
        materialized = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_complex.materialize",
                input=case["presentation"],
            )
        )
        materialized_payload = _result_payload(
            authorized_complete_runtime, materialized
        )
        complex_ = materialized_payload["complex"]
        computed = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.compute",
                input={
                    "complex": complex_,
                    "convention": case["convention"],
                },
            )
        )
        groups = _result_payload(authorized_complete_runtime, computed)["groups"]
        assert [group["betti_number"] for group in groups] == (
            case["expected_free_ranks"]
        )
        assert [group["torsion_coefficients"] for group in groups] == (
            case["expected_torsion"]
        )
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = authorized_complete_runtime.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.verify",
                mode=CapabilityMode.VERIFY,
                input={"result_uri": computed.output["result_uri"]},
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
