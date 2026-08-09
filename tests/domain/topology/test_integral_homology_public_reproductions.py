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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = PROJECT_ROOT / "benchmarks" / "datasets" / "public-reproductions-v1"


def _suite() -> list[dict[str, Any]]:
    cases = []
    for slug in ("integral-circle", "integral-projective-plane", "reduced-point"):
        task = TASK_ROOT / slug
        request = json.loads((task / "environment" / "input.json").read_text())
        expected = json.loads((task / "tests" / "expected.json").read_text())
        cases.append(
            {
                "presentation": request["presentation"],
                "convention": request["convention"],
                "expected_free_ranks": expected["expected_free_ranks"],
                "expected_torsion": expected["expected_torsion"],
            }
        )
    return cases


def _inline_result_payload(computed) -> dict[str, Any]:
    return computed.output["result"]


def test_public_integral_homology_cases_bind_generators_and_torsion(
    topology_services,
) -> None:
    for case in _suite():
        canonicalized = topology_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_complex.canonicalize",
                input=case["presentation"],
            )
        )
        complex_ = _inline_result_payload(canonicalized)["complex"]
        integral_input = {
            "complex": complex_,
            "convention": case["convention"],
        }
        computed = topology_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.compute",
                input=integral_input,
            )
        )
        groups = _inline_result_payload(computed)["groups"]
        assert [group["betti_number"] for group in groups] == (
            case["expected_free_ranks"]
        )
        assert [group["torsion_coefficients"] for group in groups] == (
            case["expected_torsion"]
        )
        assert computed.assurance.level is CapabilityAssuranceLevel.COMPUTED

        verified = topology_services.core.capabilities.invoke(
            CapabilityRequest(
                capability_id="topology.simplicial_homology.integral.verify",
                mode=CapabilityMode.VERIFY,
                input={
                    "input": integral_input,
                    "candidate": _inline_result_payload(computed),
                },
            )
        )
        assert verified.execution.status is ExecutionStatus.COMPLETED
        assert verified.output["status"] == "VERIFIED"
        assert verified.assurance.level is CapabilityAssuranceLevel.VERIFIED
